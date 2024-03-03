# SuperFastPython.com
# example of running a function in another process
from time import sleep
from multiprocessing import Process, Value, set_start_method


#  a custom function that blocks for a moment
def task(sleep_time, message):
    # block for a moment
    sleep(sleep_time)
    # display a message
    print(message)


# custom process class
class CustomProcess(Process):

    # override the constructor
    def __init__(self):
        # execute the base constructor
        Process.__init__(self)

        # An instance of the multiprocessing.Value can be defined in the constructor of a custom class as a shared instance variable.
        # The constructor of the multiprocessing.Value class requires that we specify the data type and an initial value.
        # We can define an instance attribute as an instance of the multiprocessing.Value
        # which will automatically and correctly be shared between processes.
        # initialize integer attribute
        self.data = Value("i", 0)

    # override the run function
    def run(self):
        # block for a moment
        sleep(1)
        # store the data variable
        self.data.value = 99
        # report stored value
        print(f"Child stored: {self.data.value}")


# entry point
if __name__ == "__main__":
    # Windows (win32): spawn
    # macOS (darwin): spawn, fork, forkserver.
    # Linux (unix): spawn, fork, forkserver.
    # set the start method
    set_start_method("spawn")
    # create a process
    # process = Process(target=task, args=(1.5, "New message from another process"))
    process = CustomProcess()

    # run the process,
    # This does not start the process immediately,
    # but instead allows the operating system to schedule the function to execute as soon as possible.
    process.start()

    # report the daemon attribute
    print(
        process.daemon, process.name, process.pid, process.exitcode, process.is_alive()
    )

    # wait for the process to finish
    print("Waiting for the process...")
    # explicitly block and wait for the new process to terminate.
    process.join()
    # report the process attribute
    print(f"Parent got: {process.data.value}")
