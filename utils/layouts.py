from brainflow import BoardIds

layouts = {
    BoardIds.SYNTHETIC_BOARD.value: {
        "channels": ["channel_1", "channel_2", "channel_3", "channel_4", "channel_5", "channel_6", "channel_7",
                     "channel_8", "channel_9", "channel_10", "channel_11", "channel_12", "channel_13", "channel_14",
                     "channel_15", "channel_16"],
        "header": ["channel_1", "channel_2", "channel_3", "channel_4", "channel_5", "channel_6", "channel_7",
                   "channel_8", "channel_9", "channel_10", "channel_11", "channel_12", "channel_13", "channel_14",
                   "channel_15", "channel_16", "Trigger"],
        "eeg_start": 0,
        "eeg_end": 16
    },
    BoardIds.UNICORN_BOARD.value: {
        "channels": ["Fz", "C3", "Cz", "C4", "Pz", "PO7", "Oz", "PO8"],
        "header": ["Fz", "C3", "Cz", "C4", "Pz", "PO7", "Oz", "PO8", "AccX", "AccY", "AccZ", "GyroX", "GyroY", "GyroZ",
                   "CNT", "BAT", "VALID", "DeltaTime", "Trigger"],
        "raw": ["Fz", "C3", "Cz", "C4", "Pz", "PO7", "Oz", "PO8", "AccX", "AccY", "AccZ", "GyroX", "GyroY", "GyroZ",
                "CNT", "BAT", "VALID", "DeltaTime", "Trigger"],
        "unity": ["Fz", "C3", "Cz", "C4", "Pz", "PO7", "Oz", "PO8", "trigger", "id", "target", "nontarget", "trial",
                  "islast"],
        "slim": ["Fz", "C3", "Cz", "C4", "Pz", "PO7", "Oz", "PO8", "STI"],
        "eeg_start": 0,
        "eeg_end": 8,
    },
    BoardIds.ENOPHONE_BOARD.value: {
        "channels": ["A1", "C3", "C4", "A2"],
        "header": ["Sample", "A1", "C3", "C4", "A2", "Time", "Trigger"],
        "eeg_start": 1,
        "eeg_end": 5
    }
}
