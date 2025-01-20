import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.logger import logging

def error_message_detail(error,error_detail=sys):
    _,_,exc_tb=error_detail.exc_info()
    filename=exc_tb.tb_frame.f_code.co_filename
    error_message="Error in python script name [{0}] line number [{1}] erro message [{2}]".format(
    filename,exc_tb.tb_lineno,str(error))
    return error_message

class CustomException(Exception):
    def __init__(self,error,error_detail:sys):
        super().__init__(error)
        self.error=error_message_detail(error,error_detail=error_detail)
    
    def __str__(self):
        return self.error