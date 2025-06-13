"""
Frequency Band Configuration Panel
Allows users to enable/disable frequency bands and adjust their ranges within predefined limits.

Author: Michele Romani
"""

import logging
from PyQt5 import QtWidgets, QtCore, QtGui
from PyQt5.QtWidgets import QSlider, QLabel, QCheckBox, QVBoxLayout, QHBoxLayout, QGroupBox
from PyQt5.QtCore import pyqtSignal


class DoubleSlider(QtWidgets.QWidget):
    """
    A custom widget with two sliders for setting frequency ranges within constraints.
    """

    valueChanged = pyqtSignal(float, float)  # low_freq, high_freq

    def __init__(self, min_val=0, max_val=100, low_val=20, high_val=80, step=0.1):
        super().__init__()

        self.min_val = min_val
        self.max_val = max_val
        self.step = step
        self.scale = int(1 / step)  # Convert to integer scale for QSlider

        self.setupUI()
        self.set_values(low_val, high_val)

    def setupUI(self):
        layout = QVBoxLayout()
        layout.setSpacing(5)

        # Create sliders
        self.low_slider = QSlider(QtCore.Qt.Horizontal)
        self.high_slider = QSlider(QtCore.Qt.Horizontal)

        # Set ranges (scaled to integers)
        slider_min = int(self.min_val * self.scale)
        slider_max = int(self.max_val * self.scale)

        self.low_slider.setRange(slider_min, slider_max)
        self.high_slider.setRange(slider_min, slider_max)

        # Style the sliders
        self.low_slider.setStyleSheet("""
            QSlider::groove:horizontal {
                border: 1px solid #999999;
                height: 6px;
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #B1B1B1, stop:1 #c4c4c4);
                margin: 2px 0;
                border-radius: 3px;
            }
            QSlider::handle:horizontal {
                background: #4CAF50;
                border: 1px solid #45a049;
                width: 16px;
                margin: -5px 0;
                border-radius: 8px;
            }
            QSlider::handle:horizontal:hover {
                background: #5cbf60;
            }
        """)

        self.high_slider.setStyleSheet("""
            QSlider::groove:horizontal {
                border: 1px solid #999999;
                height: 6px;
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #B1B1B1, stop:1 #c4c4c4);
                margin: 2px 0;
                border-radius: 3px;
            }
            QSlider::handle:horizontal {
                background: #FF5722;
                border: 1px solid #e64a19;
                width: 16px;
                margin: -5px 0;
                border-radius: 8px;
            }
            QSlider::handle:horizontal:hover {
                background: #ff7043;
            }
        """)

        # Create labels with constraints info
        self.constraint_label = QLabel(f"Range: {self.min_val:.1f} - {self.max_val:.1f} Hz")
        self.constraint_label.setStyleSheet("color: #bbb; font-size: 10px; font-style: italic;")
        self.constraint_label.setAlignment(QtCore.Qt.AlignCenter)

        self.low_label = QLabel("Low: 0.0 Hz")
        self.high_label = QLabel("High: 0.0 Hz")
        self.range_label = QLabel("Selected: 0.0 - 0.0 Hz")

        # Style labels
        label_style = "color: white; font-size: 11px;"
        self.low_label.setStyleSheet(label_style)
        self.high_label.setStyleSheet(label_style)
        self.range_label.setStyleSheet("color: #4CAF50; font-weight: bold; font-size: 12px;")

        # Connect signals
        self.low_slider.valueChanged.connect(self.on_low_changed)
        self.high_slider.valueChanged.connect(self.on_high_changed)

        # Layout
        layout.addWidget(self.constraint_label)

        low_layout = QHBoxLayout()
        low_layout.addWidget(QLabel("Low"))
        low_layout.addWidget(self.low_slider, 3)
        low_layout.addWidget(self.low_label, 1)

        high_layout = QHBoxLayout()
        high_layout.addWidget(QLabel("High"))
        high_layout.addWidget(self.high_slider, 3)
        high_layout.addWidget(self.high_label, 1)

        layout.addLayout(low_layout)
        layout.addLayout(high_layout)
        layout.addWidget(self.range_label)

        self.setLayout(layout)

    def on_low_changed(self, value):
        low_val = value / self.scale
        high_val = self.high_slider.value() / self.scale

        # Ensure low <= high with minimum gap
        min_gap = 0.5  # Minimum 0.5 Hz gap
        if low_val >= high_val - min_gap:
            high_val = min(low_val + min_gap, self.max_val)
            self.high_slider.setValue(int(high_val * self.scale))

        self.update_labels()
        self.valueChanged.emit(low_val, high_val)

    def on_high_changed(self, value):
        high_val = value / self.scale
        low_val = self.low_slider.value() / self.scale

        # Ensure low <= high with minimum gap
        min_gap = 0.5  # Minimum 0.5 Hz gap
        if high_val <= low_val + min_gap:
            low_val = max(high_val - min_gap, self.min_val)
            self.low_slider.setValue(int(low_val * self.scale))

        self.update_labels()
        self.valueChanged.emit(low_val, high_val)

    def update_labels(self):
        low_val = self.low_slider.value() / self.scale
        high_val = self.high_slider.value() / self.scale

        self.low_label.setText(f"Low: {low_val:.1f} Hz")
        self.high_label.setText(f"High: {high_val:.1f} Hz")
        self.range_label.setText(f"Selected: {low_val:.1f} - {high_val:.1f} Hz")

    def set_values(self, low_val, high_val):
        """Set the slider values programmatically"""
        # Clamp values to constraints
        low_val = max(self.min_val, min(low_val, self.max_val))
        high_val = max(self.min_val, min(high_val, self.max_val))

        self.low_slider.setValue(int(low_val * self.scale))
        self.high_slider.setValue(int(high_val * self.scale))
        self.update_labels()

    def get_values(self):
        """Get current values as (low, high) tuple"""
        low_val = self.low_slider.value() / self.scale
        high_val = self.high_slider.value() / self.scale
        return (low_val, high_val)


class FrequencyBandWidget(QtWidgets.QWidget):
    """
    Widget for a single frequency band with checkbox and constrained double slider.
    """

    bandChanged = pyqtSignal(str, bool, float, float)  # band_name, enabled, low_freq, high_freq

    def __init__(self, band_name, min_freq, max_freq, current_low=None, current_high=None, enabled=True):
        super().__init__()

        self.band_name = band_name
        self.min_freq = min_freq
        self.max_freq = max_freq
        self.enabled = enabled

        # Use current values or default to min/max
        if current_low is None:
            current_low = min_freq
        if current_high is None:
            current_high = max_freq

        self.setupUI(current_low, current_high)

    def setupUI(self, current_low, current_high):
        layout = QVBoxLayout()
        layout.setSpacing(8)

        # Header with checkbox and band name
        header_layout = QHBoxLayout()

        self.checkbox = QCheckBox(f"{self.band_name.capitalize()} ({self.min_freq}-{self.max_freq} Hz)")
        self.checkbox.setChecked(self.enabled)
        self.checkbox.setStyleSheet("""
            QCheckBox {
                color: white;
                font-weight: bold;
                font-size: 14px;
            }
            QCheckBox::indicator {
                width: 18px;
                height: 18px;
            }
            QCheckBox::indicator:checked {
                background-color: #4CAF50;
                border: 2px solid #45a049;
                border-radius: 3px;
            }
            QCheckBox::indicator:unchecked {
                background-color: #555;
                border: 2px solid #777;
                border-radius: 3px;
            }
        """)

        header_layout.addWidget(self.checkbox)
        header_layout.addStretch()

        # Double slider constrained to this band's range
        self.double_slider = DoubleSlider(
            min_val=self.min_freq,
            max_val=self.max_freq,
            low_val=current_low,
            high_val=current_high,
            step=0.1
        )

        # Connect signals
        self.checkbox.stateChanged.connect(self.on_enabled_changed)
        self.double_slider.valueChanged.connect(self.on_range_changed)

        # Layout
        layout.addLayout(header_layout)
        layout.addWidget(self.double_slider)

        # Enable/disable slider based on checkbox
        self.double_slider.setEnabled(self.enabled)

        self.setLayout(layout)

    def on_enabled_changed(self, state):
        """Handle checkbox state change"""
        self.enabled = state == QtCore.Qt.Checked
        self.double_slider.setEnabled(self.enabled)

        low_freq, high_freq = self.double_slider.get_values()
        self.bandChanged.emit(self.band_name, self.enabled, low_freq, high_freq)

    def on_range_changed(self, low_freq, high_freq):
        """Handle frequency range change"""
        if self.enabled:
            self.bandChanged.emit(self.band_name, self.enabled, low_freq, high_freq)

    def get_band_config(self):
        """Get current band configuration"""
        low_freq, high_freq = self.double_slider.get_values()
        return {
            'enabled': self.enabled,
            'range': (low_freq, high_freq),
            'constraints': (self.min_freq, self.max_freq)
        }

    def set_band_config(self, enabled, low_freq, high_freq):
        """Set band configuration"""
        self.checkbox.setChecked(enabled)
        self.double_slider.set_values(low_freq, high_freq)


class FrequencyBandPanel(QtWidgets.QWidget):
    """
    Main panel for configuring frequency bands with individual constraints.
    """

    bandsChanged = pyqtSignal(dict)  # Emits the complete freq_bands dictionary
    averageChanged = pyqtSignal(bool)  # Emits the average state

    def __init__(self, band_constraints=None):
        super().__init__()

        if band_constraints is None:
            # Default band constraints (min, max for each band)
            band_constraints = {
                'delta': (1, 4),
                'theta': (4, 8),
                'alpha': (8, 12),
                'beta': (12, 30),
                'gamma': (30, 80)
            }

        self.band_constraints = band_constraints
        self.band_widgets = {}
        self.average = False
        self.setupUI()

    def setupUI(self):
        layout = QVBoxLayout()
        layout.setSpacing(10)

        # Title
        title = QLabel("Frequency Bands Configuration")
        title.setStyleSheet("color: white; font-size: 18px; font-weight: bold; margin-bottom: 10px;")
        title.setAlignment(QtCore.Qt.AlignCenter)
        layout.addWidget(title)

        # Average value checkbox
        self.average_checkbox = QCheckBox("Average Frequency Values")
        self.average_checkbox.setChecked(self.average)
        self.average_checkbox.setStyleSheet("""
            QCheckBox {
                color: white;
                font-size: 14px;
                font-weight: bold;
            }
            QCheckBox::indicator {
                width: 18px;
                height: 18px;
            }
            QCheckBox::indicator:checked {
                background-color: #4CAF50;
                border: 2px solid #45a049;
                border-radius: 3px;
            }
            QCheckBox::indicator:unchecked {
                background-color: #555;
                border: 2px solid #777;
                border-radius: 3px;
            }
        """)

        self.average_checkbox.stateChanged.connect(self.on_average_changed)
        layout.addWidget(self.average_checkbox)

        # Scroll area for bands
        scroll_area = QtWidgets.QScrollArea()
        scroll_widget = QtWidgets.QWidget()
        scroll_layout = QVBoxLayout()
        scroll_layout.setSpacing(5)

        # Create band widgets with individual constraints
        for band_name, (min_freq, max_freq) in self.band_constraints.items():
            band_widget = FrequencyBandWidget(
                band_name=band_name,
                min_freq=min_freq,
                max_freq=max_freq,
                current_low=min_freq,  # Start at min
                current_high=max_freq,  # Start at max
                enabled=True
            )
            band_widget.bandChanged.connect(self.on_band_changed)

            # Add to group box for better visual separation
            group_box = QGroupBox()
            group_box.setStyleSheet("""
                QGroupBox {
                    border: 2px solid #555;
                    border-radius: 8px;
                    margin: 2px;
                    padding: 8px;
                    background-color: #3a3a3a;
                }
            """)
            group_layout = QVBoxLayout()
            group_layout.setContentsMargins(5, 5, 5, 5)
            group_layout.addWidget(band_widget)
            group_box.setLayout(group_layout)

            scroll_layout.addWidget(group_box)
            self.band_widgets[band_name] = band_widget

        scroll_widget.setLayout(scroll_layout)
        scroll_area.setWidget(scroll_widget)
        scroll_area.setWidgetResizable(True)
        scroll_area.setMaximumHeight(400)  # Limit height for better layout

        layout.addWidget(scroll_area)

        # Control buttons
        button_layout = QHBoxLayout()

        reset_button = QtWidgets.QPushButton("Reset")
        reset_button.clicked.connect(self.reset_to_full_range)
        reset_button.setStyleSheet("""
            QPushButton {
                background-color: #ff9800;
                color: white;
                border: none;
                padding: 8px 16px;
                border-radius: 4px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #f57c00;
            }
        """)

        enable_all_button = QtWidgets.QPushButton("Enable All")
        enable_all_button.clicked.connect(self.enable_all_bands)
        enable_all_button.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
                border: none;
                padding: 8px 16px;
                border-radius: 4px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
        """)

        disable_all_button = QtWidgets.QPushButton("Disable All")
        disable_all_button.clicked.connect(self.disable_all_bands)
        disable_all_button.setStyleSheet("""
            QPushButton {
                background-color: #f44336;
                color: white;
                border: none;
                padding: 8px 16px;
                border-radius: 4px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #da190b;
            }
        """)

        button_layout.addWidget(reset_button)
        button_layout.addWidget(enable_all_button)
        button_layout.addWidget(disable_all_button)
        layout.addLayout(button_layout)

        # Status label
        self.status_label = QLabel("Ready - Configure frequency bands above")
        self.status_label.setStyleSheet("color: #4CAF50; font-weight: bold; margin-top: 10px; padding: 5px;")
        self.status_label.setAlignment(QtCore.Qt.AlignCenter)
        layout.addWidget(self.status_label)

        self.setLayout(layout)

        # Set panel properties
        self.setMinimumWidth(350)
        self.setMaximumWidth(400)
        self.setStyleSheet("background-color: #2b2b2b; color: white;")

        # Emit initial configuration
        self.bandsChanged.emit(self.get_frequency_bands())

    def on_band_changed(self, band_name, enabled, low_freq, high_freq):
        """Handle individual band changes"""
        constraints = self.band_constraints[band_name]
        logging.debug(f"Band changed: {band_name} = {enabled}, ({low_freq:.1f}, {high_freq:.1f}) Hz [constraints: {constraints[0]}-{constraints[1]}]")

        if enabled:
            self.status_label.setText(f"Modified: {band_name} ({low_freq:.1f}-{high_freq:.1f} Hz)")
        else:
            self.status_label.setText(f"Disabled: {band_name}")

        # Emit the complete configuration
        self.bandsChanged.emit(self.get_frequency_bands())

    def on_average_changed(self, state):
        """Handle average checkbox state change"""
        self.average = state == QtCore.Qt.Checked
        if self.average:
            self.status_label.setText("Average Frequency Values Enabled")
        else:
            self.status_label.setText("Average Frequency Values Disabled")

        # Emit the complete configuration
        self.averageChanged.emit(self.average)

    def get_frequency_bands(self):
        """Get the current frequency bands configuration"""
        freq_bands = {}

        for band_name, widget in self.band_widgets.items():
            config = widget.get_band_config()
            if config['enabled']:
                freq_bands[band_name] = config['range']

        return freq_bands

    def set_frequency_bands(self, freq_bands):
        """Set the frequency bands configuration"""
        for band_name, widget in self.band_widgets.items():
            if band_name in freq_bands:
                low_freq, high_freq = freq_bands[band_name]
                widget.set_band_config(True, low_freq, high_freq)
            else:
                # Band is disabled - keep current range but disable
                current_config = widget.get_band_config()
                low_freq, high_freq = current_config['range']
                widget.set_band_config(False, low_freq, high_freq)

    def reset_to_full_range(self):
        """Reset all bands to their full constraint ranges"""
        full_range_bands = {}

        for band_name, (min_freq, max_freq) in self.band_constraints.items():
            widget = self.band_widgets[band_name]
            widget.set_band_config(True, min_freq, max_freq)
            full_range_bands[band_name] = (min_freq, max_freq)

        self.status_label.setText("Reset all bands to full ranges")
        self.bandsChanged.emit(full_range_bands)

    def enable_all_bands(self):
        """Enable all frequency bands"""
        for widget in self.band_widgets.values():
            config = widget.get_band_config()
            low_freq, high_freq = config['range']
            widget.set_band_config(True, low_freq, high_freq)

        self.status_label.setText("Enabled all frequency bands")
        self.bandsChanged.emit(self.get_frequency_bands())

    def disable_all_bands(self):
        """Disable all frequency bands"""
        for widget in self.band_widgets.values():
            config = widget.get_band_config()
            low_freq, high_freq = config['range']
            widget.set_band_config(False, low_freq, high_freq)

        self.status_label.setText("Disabled all frequency bands")
        self.bandsChanged.emit(self.get_frequency_bands())


# ===================== DEMO APPLICATION =====================

class FrequencyBandDemo(QtWidgets.QMainWindow):
    """
    Demo application to test the frequency band panel with constraints.
    """

    def __init__(self):
        super().__init__()
        self.setupUI()

    def setupUI(self):
        self.setWindowTitle("Frequency Band Configuration Demo - Constrained Ranges")
        self.setGeometry(100, 100, 1000, 700)

        central_widget = QtWidgets.QWidget()
        self.setCentralWidget(central_widget)

        layout = QHBoxLayout()

        # Create frequency band panel with constraints
        band_constraints = {
            'delta': (1, 4),      # Delta can only be 1-4 Hz
            'theta': (4, 8),      # Theta can only be 4-8 Hz
            'alpha': (8, 12),     # Alpha can only be 8-12 Hz
            'beta': (12, 30),     # Beta can only be 12-30 Hz
            'gamma': (30, 50)     # Gamma can only be 30-50 Hz
        }

        self.freq_panel = FrequencyBandPanel(band_constraints)
        self.freq_panel.bandsChanged.connect(self.on_bands_changed)
        self.freq_panel.averageChanged.connect(self.on_average_changed)

        # Create output display
        self.output_display = QtWidgets.QTextEdit()
        self.output_display.setReadOnly(True)
        self.output_display.setStyleSheet("""
            QTextEdit {
                background-color: #1e1e1e;
                color: #00ff00;
                font-family: 'Courier New', monospace;
                font-size: 12px;
                border: 2px solid #555;
                padding: 10px;
            }
        """)

        # Initial display
        self.update_output_display(self.freq_panel.get_frequency_bands())

        layout.addWidget(self.freq_panel, 1)
        layout.addWidget(self.output_display, 2)  # Give more space to output

        central_widget.setLayout(layout)

        # Set dark theme
        self.setStyleSheet("background-color: #2b2b2b;")

    def on_bands_changed(self, freq_bands):
        """Handle frequency bands changes"""
        self.update_output_display(freq_bands)

    def on_average_changed(self, average):
        """Handle average checkbox state change"""
        # Update display with current configuration
        freq_bands = self.freq_panel.get_frequency_bands()
        self.update_output_display(freq_bands)
        if average:
            self.output_display.append("Average Frequency Values Enabled")
        else:
            self.output_display.append("Average Frequency Values Disabled")



    def update_output_display(self, freq_bands):
        """Update the output display with current configuration"""
        output = "freq_bands = {\n"
        for band_name, (low, high) in freq_bands.items():
            output += f"    '{band_name}': ({low:.1f}, {high:.1f}),\n"
        output += "}\n\n"

        output += f"# Enabled bands: {len(freq_bands)}\n"
        if freq_bands:
            output += f"# Frequency range: {min(low for low, high in freq_bands.values()):.1f} - {max(high for low, high in freq_bands.values()):.1f} Hz\n"

        output += "\n# Band Constraints:\n"
        output += "# delta: 1-4 Hz (can't go outside this range)\n"
        output += "# theta: 4-8 Hz (can't go outside this range)\n"
        output += "# alpha: 8-12 Hz (can't go outside this range)\n"
        output += "# beta: 12-30 Hz (can't go outside this range)\n"
        output += "# gamma: 30-50 Hz (can't go outside this range)\n"

        self.output_display.setText(output)


if __name__ == '__main__':
    import sys

    app = QtWidgets.QApplication(sys.argv)

    # Set application-wide dark theme
    app.setStyle('Fusion')
    dark_palette = QtGui.QPalette()
    dark_palette.setColor(QtGui.QPalette.Window, QtGui.QColor(53, 53, 53))
    dark_palette.setColor(QtGui.QPalette.WindowText, QtGui.QColor(255, 255, 255))
    app.setPalette(dark_palette)

    demo = FrequencyBandDemo()
    demo.show()

    sys.exit(app.exec_())