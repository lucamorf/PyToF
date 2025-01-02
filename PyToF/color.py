class c:

    """
    This class is solely used to colour the verbosity output in the console.
    """

    WARN = '\033[91m'
    GOOD = '\033[92m'
    SPEC = '\033[93m'
    INFO = '\033[94m'
    NUMB = '\033[96m'
    ENDC = '\033[0m'

    def get(bool):

        if bool:

            return c.GOOD

        else:

            return c.WARN