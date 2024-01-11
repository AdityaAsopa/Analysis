'''Exception definitions related to the analysis'''


class EIDynamicsModuleError(Exception):
    pass


class FileMismatchError(EIDynamicsModuleError):
    '''Error raised when the expected file is not found'''
    def __init__(self):
        # Call Exception.__init__(message)
        # to use the same Message header as the parent class
        message = "Datafile field in expt parameter file must match the datafile name supplied."
        error = "Datafile mismatch!"
        super().__init__(message)
        self.error = error
        print("The following error was encountered: ", error)


class ParameterMismatchError(EIDynamicsModuleError):
    '''Error raised when the experiment parameters do not match the recorded data'''
    def __init__(self,message="msg",error="err"):
        # Call Exception.__init__(message)
        # to use the same Message header as the parent class
        if message == "msg":
            message = "Experiment Parameter(s) do not match the data."
        if error == "err":
            error = "Parameter mismatch!"
        super().__init__(message)
        self.error = error
        print("The following error was encountered: ", error)
