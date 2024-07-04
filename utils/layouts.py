from brainflow import BoardIds

layouts = {
    BoardIds.SYNTHETIC_BOARD.value: {
        "channels": ["Fp1", "Fp2", "F7", "F3", "Fz", "F4", "F8", "T3",
                     "C3", "Cz", "C4", "T4", "T5", "P3", "Pz", "P4"],
        "header": ["Sample","Fp1", "Fp2", "F7", "F3", "Fz", "F4", "F8", "T3",
                   "C3", "Cz", "C4", "T4", "T5", "P3", "Pz", "P4", "ch18", "ch19", "ch20", "ch21", "ch22", "ch23",
                   "ch24", "ch25", "ch26", "ch27", "ch28", "ch29", "ch30", "Time", "Trigger"],
        "eeg_start": 1,
        "eeg_end": 17
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
    BoardIds.ANT_NEURO_EE_411_BOARD.value: {
        "channels": ["Fpz", "F3", "Fz", "F4", "C3", "Cz", "C4", "Pz"],
        "header": ["Sample", "Fpz", "F3", "Fz", "F4", "C3", "Cz", "C4", "Pz", "id", "Time", "Trigger"],
        "eeg_start": 1,
        "eeg_end": 9,
    },
    BoardIds.ENOPHONE_BOARD.value: {
        "channels": ["A1", "C3", "C4", "A2"],
        "header": ["Sample", "A1", "C3", "C4", "A2", "Time", "Trigger"],
        "eeg_start": 1,
        "eeg_end": 5
    }
}
