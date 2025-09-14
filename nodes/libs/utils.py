import logging,os,sys,copy,urllib,tqdm

class Options:
    img2img_background_color = "#ffffff"  # Set to white for now

class State:
    interrupted = False
    def begin(self):
        pass
    def end(self):
        pass

class shared:
    def __init__(self):
        self.opts = Options()
        self.state = State()
        self.cmd_opts = None
        self.sd_upscalers = []
        self.face_restorers = []
def download(url, path, name):
    request = urllib.request.urlopen(url)
    total = int(request.headers.get('Content-Length', 0))
    with tqdm(total=total, desc=f'[ReActor] Downloading {name} to {path}', unit='B', unit_scale=True, unit_divisor=1024) as progress:
        urllib.request.urlretrieve(url, path, reporthook=lambda count, block_size, total_size: progress.update(block_size))
def move_path(old_path, new_path):
    if os.path.exists(old_path):
        try:
            models = os.listdir(old_path)
            for model in models:
                move_old_path = os.path.join(old_path, model)
                move_new_path = os.path.join(new_path, model)
                os.rename(move_old_path, move_new_path)
            os.rmdir(old_path)
        except Exception as e:
            print(f"Error: {e}")
            new_path = old_path
def addLoggingLevel(levelName, levelNum, methodName=None):
    if not methodName:
        methodName = levelName.lower()

    def logForLevel(self, message, *args, **kwargs):
        if self.isEnabledFor(levelNum):
            self._log(levelNum, message, args, **kwargs)

    def logToRoot(message, *args, **kwargs):
        logging.log(levelNum, message, *args, **kwargs)

    logging.addLevelName(levelNum, levelName)
    setattr(logging, levelName, levelNum)
    setattr(logging.getLoggerClass(), methodName, logForLevel)
    setattr(logging, methodName, logToRoot)

class ColoredFormatter(logging.Formatter):
    COLORS = {
        "DEBUG": "\033[0;36m",  # CYAN
        "STATUS": "\033[38;5;173m",  # Calm ORANGE
        "INFO": "\033[0;32m",  # GREEN
        "WARNING": "\033[0;33m",  # YELLOW
        "ERROR": "\033[0;31m",  # RED
        "CRITICAL": "\033[0;37;41m",  # WHITE ON RED
        "RESET": "\033[0m",  # RESET COLOR
    }

    def format(self, record):
        colored_record = copy.copy(record)
        levelname = colored_record.levelname
        seq = self.COLORS.get(levelname, self.COLORS["RESET"])
        colored_record.levelname = f"{seq}{levelname}{self.COLORS['RESET']}"
        return super().format(colored_record)


# Create a new logger
logger = logging.getLogger("ReActor")
logger.propagate = False

# Add Custom Level
# logging.addLevelName(logging.INFO, "STATUS")
addLoggingLevel("STATUS", logging.INFO + 5)

# Add handler if we don't have one.
if not logger.handlers:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(
        ColoredFormatter("[%(name)s] %(asctime)s - %(levelname)s - %(message)s",datefmt="%H:%M:%S")
    )
    logger.addHandler(handler)

# Configure logger
loglevel_string = getattr(shared().cmd_opts, "reactor_loglevel", "INFO")
loglevel = getattr(logging, loglevel_string.upper(), "info")
logger.setLevel(loglevel)

class MyLogger:
    def __init__(self, name):
        self.logger = logging.getLogger(name)
        self.logger.propagate = False
        addLoggingLevel("STATUS", logging.INFO + 5)
        # Add handler if we don't have one.
        if not logger.handlers:
            handler = logging.StreamHandler(sys.stdout)
            handler.setFormatter(
                ColoredFormatter("[%(name)s] %(asctime)s - %(levelname)s - %(message)s",datefmt="%H:%M:%S")
            )
            logger.addHandler(handler)
        # Configure logger
        loglevel_string = getattr(shared().cmd_opts, "reactor_loglevel", "INFO")
        loglevel = getattr(logging, loglevel_string.upper(), "info")
        logger.setLevel(loglevel)

    