from data_preprocessing import post_process_statbot_json
from build_model import *
from databot import start_chat


post_process_statbot_json()
build_statbot()
start_chat()

