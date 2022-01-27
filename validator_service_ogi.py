import os
import gc
import sys
import pdb
import threading
from time import sleep

# add paths to project modules
current_relative = os.path.dirname(__file__)
path_previous = ".."
sys.path.append(os.path.join(current_relative, path_previous))
from module_appender import add_paths
add_paths()

# Project modules
import global_params as glp
from common import timer, get_time

from sender import MessageSender
from listener import MessageListener
from MLCV_communication import MLCV_Communication

from logger import LogLevel
from message import Message
from log_messages import LogMessages

from db_tables import DBTables
from db_connector import DBDataConnection
from db_controls import select_from_table, db_insert_query, db_update_query

from validator import MlrrValidator
from MLCB_validator import MLCBValidator

from model_loading import get_latest_model
from config_parser import read_configuration, \
                          read_configuration_param, \
                          remove_hashed_config, \
                          get_communication_params
from exceptions import ConfigurationMissingException, \
                       UnableToConnectDBException, \
                       ServiceRestartException, \
                       CriticalComponentMissing



class ValidatorService:
    """
    A Validator Service. Listenes for a parsed data comming from DataParserService..

    Attributes:
        db_connection: object of type DBDataConnection. Provides access to the GreenPlum DB.
    """

    def __init__(self, dbConnection):
        """
        Init ParserService object
        """
        self.msg_broker = MessageSender()
        self.listeners_pool = []

        try:
            self.data_header_separatar = "data$%&_HEADsep"
            self.db_connection = dbConnection
            self._service_params = read_configuration("ValidationService")
            self.communication_params = get_communication_params()
            # self._kafka_params = read_configuration(section_name = "Kafka", db_connection = self.db_connection)
            self._enabled_projects = read_configuration_param("Main", "EnabledProjects").split(',')
        except UnableToConnectDBException as err_db:
            raise err_db
        except ConfigurationMissingException as err_cnf:
            raise err_cnf

        print("Loading Models from DB...")
        print("Loading scaler...")

        if "MLRR" in self._enabled_projects:
            try:
                scaler, model_id = get_latest_model("scaler")
            except Exception as err:
                raise CriticalComponentMissing(err)

            print("Loading classifier...")
            try:
                classifier, model_id = get_latest_model("classifier")
            except Exception as err:
                raise CriticalComponentMissing(err)

            try:
                label_classes_object, model_id = get_latest_model("label_encoder")
            except Exception as err:
                raise CriticalComponentMissing(err)

            # Init Validator
            try:
                self.validator = MlrrValidator(scaler, classifier, label_classes_object, self.db_connection)
            except Exception as ex:
                # TODO: write this in DB
                pass
        if "MLCB" in self._enabled_projects:
            print("Loading classifier...")
            try:
                classification_model, model_id = get_latest_model("MLCB_classif", )
            except Exception as err:
                raise CriticalComponentMissing(err)

            print("Loading labels...")
            try:
                labels, model_id = get_latest_model("MLCB_label_e")
            except Exception as err:
                raise CriticalComponentMissing(err)

        # Init Validator
            try:
                self.mlcb_validator = MLCBValidator(classification_model, labels)
            except Exception as ex:
                # TODO: write this in DB
                pass

        self._start_heartbeat()
        self._run()


    def _get_correct_order_list(self, unordered):
        correct_order = []

        for i in range(1, len(unordered) + 1, 1):
            correct_order.append(unordered.index(i))
        return correct_order


    def handler_restart_event(self, MessageListener, message, project_name: str, kafka_topic: str = None):
        raise ServiceRestartException()


    def handler_restart_event_on_new_model(self, MessageListener, message, project_name: str, kafka_topic: str = None):
        """
        This event handler is triggered when a 'restart' message was received on Kafka/Rabbit MQ queue "QUEUE_CONFIGURATION_UPDATE".
        Usually such message arrives when a new model is created and the service needs to be restarted.
        """
        if message is not None:
            try:
                glp.restartRequest = True
            except:
                pass


    def handler_message_event(self, MessageListener, message, project_name: str, kafka_topic: str = None, msg_original: str = None):
        print("Received request. Processing....")
        print("Incident Ids: " + str(message["incident_ids"]))
        if message["project_name"] in self._enabled_projects:
            if message["project_name"] == "MLRR":
                # TODO: parse message - it can contain multiple incident meta's. For now -assume single message
                ids = message["incident_ids"].split(',')
                vector_info = message["features"]
                vectors = [float(y.strip()) for y in str([x for x in vector_info]).strip('[]').split(',')]
                self.validator.run_validator(ids, vectors)

                # Status Update per incident
                for incident_id in ids:
                    incident_id = incident_id.strip("'")
                    # print("Service ID: " + str(incident_id))
                    dbtable_status_code__dict = {
                        "incident_id": incident_id,
                        "status_code_id": 3
                    }
                    query_status_insert_status, _ = db_insert_query(DBTables.MLRR_INCIDENT_STATUS, dbtable_status_code__dict)
                    self.db_connection.add_query(query_status_insert_status)
                    query_status_insert_status_in_time, _ = db_insert_query(DBTables.MLRR_INCIDENT_STATUS_IN_TIME, dbtable_status_code__dict)
                    self.db_connection.add_query(query_status_insert_status_in_time)

            elif message["project_name"] == "MLCB":
                ids = message["incident_ids"].split(',')
                features_info = message["features"]
                features = [float(y.strip()) for y in str([x for x in features_info]).strip('[]').split(',')]
                result_json = self.mlcb_validator.main(features, threshold = 0.1, N_Top_Results = 3)
                
                try:
                    dict_validation_result = {'results_list': result_json}
                    query, validation_id = db_insert_query(DBTables.MLRR_VALIDATION_RESULT, dict_validation_result)
                    self.db_connection.add_query(query)
                except Exception as err:
                    print("Error 1: " + str(err))
                incident_id = str(ids[0]).strip("'")
                try:
                    dict_validation_id = {'validation_id': validation_id}
                    dict_incident_id = {'usecase_search_id': incident_id}

                    query = db_update_query(DBTables.MLCB_USECASE_SEARCH_DATA, dict_validation_id, dict_incident_id)
                    self.db_connection.add_query(query)
                except Exception as err:
                    print("Error 2: " + str(err))

            else:
                # other project types
                # print("Unrecognized project type: " + str(project_name))
                # TODO: log error
                pass
        
        MessageListener.unqueue_message(msg_original)
        

    def _start_heartbeat(self):
        # Start heartbeat
        print("NEW THREAD created: " + "validator, 215")
        thread = threading.Thread(target = self._heartbeat)
        thread.daemon = True
        thread.start()


    def _heartbeat(self):
        heartbeat_interval = 2
        msg_expiration_sec = 60

        while True:
            beat = int(get_time())
            beat_msg = '{ "service_id": "' + glp.service_id + '", "service_type": "data_validator", "heartbeat": ' + str(beat) + ' }'
            # if uuid in tcpConnectors:
            self.msg_broker.send(
                message_object = Message(beat_msg),
                queue_topic = self.communication_params.QUEUE_HEARTBEATS,
                expiration = msg_expiration_sec)
            if glp.restartRequest:
                break
            sleep(heartbeat_interval)


    def _run(self):
        """
        Main method for the service.
        """
        # define a listner - for configuration update
        config_listener = MessageListener(project_name="serviceValidator", queue_topic=self.communication_params.QUEUE_CONFIGURATION_UPDATE)
        config_listener.Listen()
        config_listener.received_message_event += self.handler_restart_event

        # define a listner - for all kafka messages between reader and parser
        queue_listener = MessageListener(project_name="", queue_topic=self.communication_params.QUEUE_PARSER_VALIDATOR)
        queue_listener.Listen()
        queue_listener.received_message_event += self.handler_message_event

        # define a listner - for new model event update
        new_model_listener = MessageListener(project_name="serviceValidator", queue_topic="msg_queue_" + glp.service_id)
        new_model_listener.Listen()
        new_model_listener.received_message_event += self.handler_restart_event_on_new_model

        self.listeners_pool.append(config_listener)
        self.listeners_pool.append(new_model_listener)
        self.listeners_pool.append(queue_listener)

        # process data
        while True:
            if config_listener.restart_required or glp.restartRequest:
                print("Restart ...")
                raise ServiceRestartException()
            sleep(1)


def tcp_message_received(_, msg_type, data):
    if msg_type == 5:
        glp.logger.log_message(LogMessages.MLCV_DB_CONFIG_SET.value, LogLevel.INFO)
        # Set DB Configuration
        with open(".dbConfig", 'w') as dbFileConfig:
            dbFileConfig.write(data[43:].decode())

    elif msg_type == 6:
        glp.logger.log_message(LogMessages.SERVICE_RESTARTED_REQUEST.value, LogLevel.INFO)
        glp.restartRequest = True
    pass


def restart_program():
    """Restarts the current program, with file objects and descriptors
       cleanup
    """

    try:
        p = psutil.Process(os.getpid())
        for handler in p.connections():
            os.close(handler.fd)
    except Exception as e:
        # TODO: log this in DB
        pass
        # print("exc: " + str(e))

    python = sys.executable
    os.execl(python, python, __file__)


# SERVICE START
if __name__ == '__main__':
    # Run forever
    while True:
        glp.init_global("Validator")
        timer()
        current_service = None
        glp.logger.log_message(LogMessages.SERVICE_STARTED.value, LogLevel.INFO)

        print("service started: " + glp.service_id)
        remove_hashed_config()

        # define Discovery listener to Wait for MLCV connection
        mlcv_connector = MLCV_Communication(service_type = 3, service_id = glp.service_id)
        mlcv_connector.message_received_event += tcp_message_received

        # Run until a Restart Request is received
        while True:
            try:
                # create a DB Connection
                dbConnection = DBDataConnection()
                current_service = ValidatorService(dbConnection)
                
            except ServiceRestartException:
                glp.logger.log_message(LogMessages.SERVICE_RESTARTED_REQUEST.value, LogLevel.INFO)
                
            except Exception as error:
                try:
                    # Log error
                    glp.logger.log_message(str(error), LogLevel.ERROR)
                except Exception:
                    pass
                # Wait N second and run the service again.
                sleep(1)
            finally:
                dbConnection.close()
                if current_service is not None:
                    for listen_er in current_service.listeners_pool:
                        listen_er.close_listener()
                    del current_service
                gc.collect()
                if glp.restartRequest:
                    glp.close_global()
                    if dbConnection is not None:
                        dbConnection.save_state()
                    glp.restartRequest = False
                    restart_program()
                    break
