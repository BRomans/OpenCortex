import logging
import time
import pylsl

"""
This script is used to simulate an LSL stream that sends markers every 250 ms.
"""


def send_test_marker(marker, interval=250):
    outlet.push_sample([marker])
    logging.info(f"Sent marker {marker}")
    time.sleep(interval / 1000)


def simulate_lsl_client(start_cmd, end_cmd, nclasses=4, duration_s=30, interval_ms=250):
    send_test_marker(start_cmd)
    time.sleep(1)
    for i in range(duration_s):
        for i in range(1, nclasses + 1):
            send_test_marker(str(i))
    time.sleep(1)
    send_test_marker(end_cmd)




if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    outlet = pylsl.StreamOutlet(pylsl.StreamInfo('Markers', 'Markers', 1, 0, 'string', 'marker'))

    cmd = ""
    while cmd.lower() is not "q":
        cmd = input("Enter T to simulate training, A to simulate application, Q to quit: ")
        if cmd.lower() == "t":
            simulate_lsl_client(start_cmd="98", end_cmd="99", nclasses=4, duration_s=5, interval_ms=250)
        elif cmd.lower() == "a":
            simulate_lsl_client(start_cmd="100", end_cmd="101", nclasses=4, duration_s=5, interval_ms=250)
