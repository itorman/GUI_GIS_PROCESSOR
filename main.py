#!/usr/bin/env python3
"""
GIS Document Processing Application
Main entry point with PyQt5 GUI for document processing and address extraction
"""

import sys
import os
from pathlib import Path
import pandas as pd
from PyQt5.QtWidgets import (QApplication, QMainWindow, QVBoxLayout, QHBoxLayout,
                             QWidget, QPushButton, QTableWidget, QTableWidgetItem,
                             QFileDialog, QLabel, QProgressBar, QMessageBox,
                             QHeaderView, QTextEdit, QTabWidget, QGroupBox,
                             QComboBox, QLineEdit, QSpinBox, QDoubleSpinBox, QCheckBox)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer, QCoreApplication
from PyQt5.QtGui import QFont, QIcon

# Import our custom modules
from preprocessing.document_processor import DocumentProcessor
from llm.llm_client import LLMClient
from postprocessing.data_processor import DataProcessor
from export.data_exporter import DataExporter
from export.standardized_exporter import StandardizedExporter
from utils.quality_assessor import QualityAssessor


class ProcessingThread(QThread):
    """Background thread for document processing"""
    progress = pyqtSignal(int)
    status = pyqtSignal(str)
    finished = pyqtSignal(object)
    error = pyqtSignal(str)

    def __init__(self, document_path, llm_config):
        super().__init__()
        self.document_path = document_path
        self.llm_config = llm_config

    def run(self):
        try:
            self.status.emit("Loading document...")
            self.progress.emit(10)
            
            # Process document
            doc_processor = DocumentProcessor()
            text_chunks = doc_processor.process_document(self.document_path)
            
            self.status.emit("Extracting addresses with LLM...")
            self.progress.emit(30)
            
            # Process with LLM using enhanced contextual processing
            llm_client = LLMClient(self.llm_config)
            
            # Use contextual processing if enabled
            if self.llm_config.get('use_contextual', True) and hasattr(llm_client, 'extract_addresses_contextual'):
                self.status.emit("Processing document with contextual address inference...")
                results = llm_client.extract_addresses_contextual(text_chunks)
                # Convert StandardizedExtraction objects to dictionaries for compatibility
                if results and hasattr(results[0], '__dict__'):
                    results = [result.__dict__ for result in results]
                self.progress.emit(70)
            elif hasattr(llm_client, 'extract_addresses_batch'):
                self.status.emit("Processing all chunks with improved batch extraction...")
                results = llm_client.extract_addresses_batch(text_chunks)
                self.progress.emit(70)
            else:
                # Fallback to individual chunk processing
                self.status.emit("Processing chunks individually...")
                results = []
                
                for i, chunk in enumerate(text_chunks):
                    self.status.emit(f"Processing chunk {i+1}/{len(text_chunks)}...")
                    chunk_result = llm_client.extract_addresses(chunk)
                    
                    # Debug logging
                    if chunk_result:
                        self.status.emit(f"Chunk {i+1}: Found {len(chunk_result)} addresses")
                        results.extend(chunk_result)
                    else:
                        self.status.emit(f"Chunk {i+1}: No addresses found")
                    
                    self.progress.emit(30 + int(40 * (i + 1) / len(text_chunks)))
            
            self.status.emit("Post-processing data...")
            self.progress.emit(80)
            
            # Post-process results
            data_processor = DataProcessor()
            processed_data = data_processor.process_results(results)
            
            self.progress.emit(100)
            self.status.emit("Processing complete!")
            self.finished.emit(processed_data)
            
        except Exception as e:
            self.error.emit(str(e))


class MainWindow(QMainWindow):
    """Main application window"""
    
    def __init__(self):
        super().__init__()
        # Initialize variables
        self.document_path = None
        self.processing_thread = None
        self.current_results = None
        
        # Setup UI
        self.setup_ui()
        self.setup_connections()
        
    def setup_ui(self):
        """Initialize the user interface"""
        self.setWindowTitle("GIS Document Processing - Address Extraction")
        self.setGeometry(100, 100, 1200, 800)
        
        # Create central widget and main layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        
        # Create tab widget
        tab_widget = QTabWidget()
        main_layout.addWidget(tab_widget)
        
        # Main processing tab
        main_tab = self.create_main_tab()
        tab_widget.addTab(main_tab, "Document Processing")
        
        # Settings tab
        settings_tab = self.create_settings_tab()
        tab_widget.addTab(settings_tab, "Settings")
        
        # Results tab
        results_tab = self.create_results_tab()
        tab_widget.addTab(results_tab, "Results")
        
        # Map visualization tab
        map_tab = self.create_map_tab()
        tab_widget.addTab(map_tab, "Map View")
        
        # Status bar
        self.status_bar = self.statusBar()
        self.status_bar.showMessage("Ready")
        
        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        main_layout.addWidget(self.progress_bar)
        
    def create_main_tab(self):
        """Create the main document processing tab"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # Document upload section
        upload_group = QGroupBox("Document Upload")
        upload_layout = QVBoxLayout(upload_group)
        
        # File selection
        file_layout = QHBoxLayout()
        self.file_label = QLabel("No file selected")
        self.file_label.setStyleSheet("border: 1px solid #ccc; padding: 5px; background: #f9f9f9;")
        file_layout.addWidget(self.file_label, 1)
        
        self.upload_btn = QPushButton("Upload Document")
        self.upload_btn.clicked.connect(self.upload_document)
        file_layout.addWidget(self.upload_btn)
        
        upload_layout.addLayout(file_layout)
        
        # Supported formats info
        formats_label = QLabel("Supported formats: PDF, Word (.docx), Excel (.xlsx), Text (.txt)")
        formats_label.setStyleSheet("color: #666; font-size: 11px;")
        upload_layout.addWidget(formats_label)
        
        layout.addWidget(upload_group)
        
        # Processing section
        process_group = QGroupBox("Processing")
        process_layout = QVBoxLayout(process_group)
        
        # Process button
        self.process_btn = QPushButton("Run Extraction")
        self.process_btn.clicked.connect(self.run_extraction)
        self.process_btn.setEnabled(False)
        self.process_btn.setStyleSheet("QPushButton { background-color: #4CAF50; color: white; padding: 10px; font-size: 14px; }")
        process_layout.addWidget(self.process_btn)
        
        # Status display
        self.status_display = QTextEdit()
        self.status_display.setMaximumHeight(100)
        self.status_display.setReadOnly(True)
        process_layout.addWidget(self.status_display)
        
        layout.addWidget(process_group)
        
        # Export section
        export_group = QGroupBox("Export")
        export_layout = QHBoxLayout(export_group)
        
        self.export_csv_btn = QPushButton("Export to CSV")
        self.export_csv_btn.clicked.connect(lambda: self.export_data('csv'))
        self.export_csv_btn.setEnabled(False)
        export_layout.addWidget(self.export_csv_btn)
        
        self.export_excel_btn = QPushButton("Export to Excel")
        self.export_excel_btn.clicked.connect(lambda: self.export_data('excel'))
        self.export_excel_btn.setEnabled(False)
        export_layout.addWidget(self.export_excel_btn)
        
        self.export_shapefile_btn = QPushButton("Export to Shapefile")
        self.export_shapefile_btn.clicked.connect(lambda: self.export_data('shapefile'))
        self.export_shapefile_btn.setEnabled(False)
        export_layout.addWidget(self.export_shapefile_btn)
        
        self.export_arcgis_btn = QPushButton("Export to ArcGIS")
        self.export_arcgis_btn.clicked.connect(lambda: self.export_data('arcgis'))
        self.export_arcgis_btn.setEnabled(False)
        export_layout.addWidget(self.export_arcgis_btn)
        
        layout.addWidget(export_group)
        
        return tab
    
    def create_settings_tab(self):
        """Create the settings configuration tab"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # Debug: Add a label to confirm the tab is working
        debug_label = QLabel("Settings Tab - LLM Configuration")
        debug_label.setStyleSheet("font-size: 16px; font-weight: bold; color: #333; padding: 10px;")
        layout.addWidget(debug_label)
        
        # LLM Configuration
        llm_group = QGroupBox("LLM Configuration")
        llm_layout = QVBoxLayout(llm_group)
        
        # LLM Type selection
        llm_type_layout = QHBoxLayout()
        llm_type_layout.addWidget(QLabel("LLM Type:"))
        self.llm_type_combo = QComboBox()
        self.llm_type_combo.addItems(["Test Mode", "Ollama", "vLLM", "OpenAI", "Local Model"])
        self.llm_type_combo.setCurrentText("Ollama")  # Set default
        
        # Debug: Add logging for ComboBox creation
        print(f"DEBUG: ComboBox created with {self.llm_type_combo.count()} items")
        print(f"DEBUG: Current ComboBox text: {self.llm_type_combo.currentText()}")
        
        llm_layout.addLayout(llm_type_layout)
        
        # Server URL
        server_layout = QHBoxLayout()
        server_layout.addWidget(QLabel("Server URL:"))
        self.server_url_edit = QLineEdit("http://localhost:11434")  # Default Ollama
        self.server_url_edit.setPlaceholderText("Enter server URL (e.g., http://localhost:11434)")
        self.server_url_edit.setEnabled(True)
        llm_layout.addLayout(server_layout)
        
        # Model name
        model_layout = QHBoxLayout()
        model_layout.addWidget(QLabel("Model:"))
        self.model_name_edit = QLineEdit("llama3:8b")
        self.model_name_edit.setPlaceholderText("Enter model name (e.g., llama3:8b)")
        self.model_name_edit.setEnabled(True)
        llm_layout.addLayout(model_layout)
        
        layout.addWidget(llm_group)
        
        # Processing Options
        options_group = QGroupBox("Processing Options")
        options_layout = QVBoxLayout(options_group)
        
        # Chunk size
        chunk_layout = QHBoxLayout()
        chunk_layout.addWidget(QLabel("Text Chunk Size:"))
        self.chunk_size_spin = QSpinBox()
        self.chunk_size_spin.setRange(100, 5000)
        self.chunk_size_spin.setValue(1000)
        chunk_layout.addWidget(self.chunk_size_spin)
        options_layout.addLayout(chunk_layout)
        
        # Enable OCR
        self.enable_ocr_check = QCheckBox("Enable OCR for scanned PDFs")
        self.enable_ocr_check.setChecked(False)
        options_layout.addWidget(self.enable_ocr_check)
        
        layout.addWidget(options_group)
        
        # Enhanced Extraction Options
        extraction_group = QGroupBox("Enhanced Extraction Settings")
        extraction_layout = QVBoxLayout(extraction_group)
        
        # Extraction mode
        mode_layout = QHBoxLayout()
        mode_layout.addWidget(QLabel("Extraction Mode:"))
        self.extraction_mode_combo = QComboBox()
        self.extraction_mode_combo.addItems([
            "Standard", 
            "High Context", 
            "Conservative",
            "Aggressive Inference"
        ])
        self.extraction_mode_combo.setCurrentText("High Context")
        self.extraction_mode_combo.setToolTip("High Context: Uses full document context for better address inference")
        mode_layout.addWidget(self.extraction_mode_combo)
        extraction_layout.addLayout(mode_layout)
        
        # Context size
        context_layout = QHBoxLayout()
        context_layout.addWidget(QLabel("Context Size:"))
        self.context_size_spin = QSpinBox()
        self.context_size_spin.setRange(500, 5000)
        self.context_size_spin.setValue(2000)
        self.context_size_spin.setToolTip("Number of characters to use as document context")
        context_layout.addWidget(self.context_size_spin)
        extraction_layout.addLayout(context_layout)
        
        # Minimum confidence
        confidence_layout = QHBoxLayout()
        confidence_layout.addWidget(QLabel("Minimum Confidence:"))
        self.min_confidence_spin = QDoubleSpinBox()
        self.min_confidence_spin.setRange(0.0, 1.0)
        self.min_confidence_spin.setSingleStep(0.1)
        self.min_confidence_spin.setValue(0.5)
        self.min_confidence_spin.setDecimals(1)
        self.min_confidence_spin.setToolTip("Filter out extractions below this confidence level")
        confidence_layout.addWidget(self.min_confidence_spin)
        extraction_layout.addLayout(confidence_layout)
        
        # Use contextual extraction
        self.use_contextual_check = QCheckBox("Enable Contextual Address Inference")
        self.use_contextual_check.setChecked(True)
        self.use_contextual_check.setToolTip("Use document context to infer complete addresses from partial information")
        extraction_layout.addWidget(self.use_contextual_check)
        
        layout.addWidget(extraction_group)
        
        # Apply Settings Button
        apply_layout = QHBoxLayout()
        self.apply_settings_btn = QPushButton("Apply Settings")
        self.apply_settings_btn.clicked.connect(self.apply_settings)
        self.apply_settings_btn.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
                border: none;
                padding: 10px 20px;
                border-radius: 5px;
                font-size: 14px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
        """)
        apply_layout.addWidget(self.apply_settings_btn)
        
        # Status label for settings
        self.settings_status_label = QLabel("Settings ready to apply")
        self.settings_status_label.setStyleSheet("color: #666; font-size: 12px; padding: 5px;")
        apply_layout.addWidget(self.settings_status_label)
        
        layout.addLayout(apply_layout)
        
        # Debug: Verify all widgets are created
        self._verify_settings_widgets()
        
        return tab
    
    def _verify_settings_widgets(self):
        """Verify that all settings widgets are created correctly"""
        try:
            print("DEBUG: Verifying settings widgets...")
            
            # Check ComboBox
            if hasattr(self, 'llm_type_combo'):
                print(f"DEBUG: ComboBox exists with {self.llm_type_combo.count()} items")
                print(f"DEBUG: ComboBox current text: {self.llm_type_combo.currentText()}")
                print(f"DEBUG: ComboBox is enabled: {self.llm_type_combo.isEnabled()}")
                print(f"DEBUG: ComboBox is visible: {self.llm_type_combo.isVisible()}")
            else:
                print("ERROR: ComboBox not found!")
            
            # Check LineEdits
            if hasattr(self, 'server_url_edit'):
                print(f"DEBUG: Server URL edit exists, text: {self.server_url_edit.text()}")
                print(f"DEBUG: Server URL edit is enabled: {self.server_url_edit.isEnabled()}")
            else:
                print("ERROR: Server URL edit not found!")
                
            if hasattr(self, 'model_name_edit'):
                print(f"DEBUG: Model name edit exists, text: {self.model_name_edit.text()}")
                print(f"DEBUG: Model name edit is enabled: {self.model_name_edit.isEnabled()}")
            else:
                print("ERROR: Model name edit not found!")
                
            print("DEBUG: Settings widgets verification complete")
            
        except Exception as e:
            print(f"ERROR in _verify_settings_widgets: {e}")
    
    def apply_settings(self):
        """Apply the current settings and update status"""
        try:
            # Get current settings
            llm_type = self.llm_type_combo.currentText()
            server_url = self.server_url_edit.text().strip()
            model = self.model_name_edit.text().strip()
            chunk_size = self.chunk_size_spin.value()
            enable_ocr = self.enable_ocr_check.isChecked()
            
            # Get enhanced extraction settings
            extraction_mode = self.extraction_mode_combo.currentText().lower().replace(' ', '_')
            context_size = self.context_size_spin.value()
            min_confidence = self.min_confidence_spin.value()
            use_contextual = self.use_contextual_check.isChecked()
            
            # Validate settings
            if not server_url:
                QMessageBox.warning(self, "Warning", "Server URL cannot be empty")
                return
            
            if not model:
                QMessageBox.warning(self, "Warning", "Model name cannot be empty")
                return
            
            # Update status
            mode_display = "Contextual" if use_contextual else "Standard"
            self.settings_status_label.setText(f"Settings applied: {llm_type} - {model} ({mode_display})")
            self.settings_status_label.setStyleSheet("color: #4CAF50; font-size: 12px; padding: 5px; font-weight: bold;")
            
            # Log settings
            self.log_status(f"Settings applied: LLM={llm_type}, Server={server_url}, Model={model}, Mode={extraction_mode}")
            
            QMessageBox.information(
                self,
                "Settings Applied",
                f"Configuration updated successfully!\n\n"
                f"LLM Type: {llm_type}\n"
                f"Server URL: {server_url}\n"
                f"Model: {model}\n"
                f"Chunk Size: {chunk_size}\n"
                f"OCR Enabled: {enable_ocr}\n"
                f"Extraction Mode: {extraction_mode.replace('_', ' ').title()}\n"
                f"Context Size: {context_size}\n"
                f"Min Confidence: {min_confidence}\n"
                f"Contextual Inference: {use_contextual}"
            )
            
        except Exception as e:
            self.settings_status_label.setText("Error applying settings")
            self.settings_status_label.setStyleSheet("color: #f44336; font-size: 12px; padding: 5px; font-weight: bold;")
            QMessageBox.critical(self, "Error", f"Failed to apply settings: {e}")
    
    def create_results_tab(self):
        """Create the results display tab"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # Results table with standardized format
        self.results_table = QTableWidget()
        self.results_table.setColumnCount(6)
        self.results_table.setHorizontalHeaderLabels([
            "Original Text", "Inferred Address", "Confidence", "Latitude DD", "Longitude DD", "Quality"
        ])
        
        # Set table properties
        header = self.results_table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.Stretch)  # Original Text
        header.setSectionResizeMode(1, QHeaderView.Stretch)  # Inferred Address
        header.setSectionResizeMode(2, QHeaderView.ResizeToContents)  # Confidence
        header.setSectionResizeMode(3, QHeaderView.ResizeToContents)  # Latitude DD
        header.setSectionResizeMode(4, QHeaderView.ResizeToContents)  # Longitude DD
        header.setSectionResizeMode(5, QHeaderView.ResizeToContents)  # Quality
        
        layout.addWidget(self.results_table)
        
        # Summary info
        self.summary_label = QLabel("No data processed yet")
        self.summary_label.setStyleSheet("padding: 10px; background: #f0f0f0; border-radius: 5px; color: #333; font-size: 13px;")
        layout.addWidget(self.summary_label)
        
        # Quality Assessment Panel
        quality_group = QGroupBox("Quality Assessment")
        quality_layout = QVBoxLayout(quality_group)
        
        self.quality_label = QLabel("No quality assessment available")
        self.quality_label.setStyleSheet("padding: 10px; color: #666; font-size: 12px;")
        quality_layout.addWidget(self.quality_label)
        
        # Quality statistics
        self.quality_stats_label = QLabel("")
        self.quality_stats_label.setStyleSheet("padding: 5px; color: #333; font-size: 11px;")
        quality_layout.addWidget(self.quality_stats_label)
        
        layout.addWidget(quality_group)
        
        return tab
    
    def upload_document(self):
        """Handle document upload"""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Document",
            "",
            "Documents (*.pdf *.docx *.xlsx *.txt);;PDF Files (*.pdf);;Word Files (*.docx);;Excel Files (*.xlsx);;Text Files (*.txt)"
        )
        
        if file_path:
            self.document_path = file_path
            self.file_label.setText(os.path.basename(file_path))
            self.process_btn.setEnabled(True)
            self.status_bar.showMessage(f"Document loaded: {os.path.basename(file_path)}")
            self.log_status(f"Document loaded: {file_path}")
    
    def run_extraction(self):
        """Run the document extraction process"""
        if not self.document_path:
            QMessageBox.warning(self, "Warning", "Please upload a document first.")
            return
        
        # Get LLM configuration including enhanced settings
        llm_config = {
            'type': self.llm_type_combo.currentText(),
            'server_url': self.server_url_edit.text(),
            'model': self.model_name_edit.text(),
            'chunk_size': self.chunk_size_spin.value(),
            'enable_ocr': self.enable_ocr_check.isChecked(),
            'extraction_mode': self.extraction_mode_combo.currentText().lower().replace(' ', '_'),
            'context_size': self.context_size_spin.value(),
            'min_confidence': self.min_confidence_spin.value(),
            'use_contextual': self.use_contextual_check.isChecked()
        }
        
        # Disable buttons during processing
        self.process_btn.setEnabled(False)
        self.upload_btn.setEnabled(False)
        
        # Show progress
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        
        # Start processing thread
        self.processing_thread = ProcessingThread(self.document_path, llm_config)
        self.processing_thread.progress.connect(self.progress_bar.setValue)
        self.processing_thread.status.connect(self.log_status)
        self.processing_thread.finished.connect(self.processing_finished)
        self.processing_thread.error.connect(self.processing_error)
        self.processing_thread.start()
    
    def log_status(self, message):
        """Log status messages"""
        from datetime import datetime
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.status_display.append(f"[{timestamp}] {message}")
        self.status_bar.showMessage(message)
    
    def processing_finished(self, results):
        """Handle processing completion with quality assessment"""
        # Re-enable buttons
        self.process_btn.setEnabled(True)
        self.upload_btn.setEnabled(True)
        self.progress_bar.setVisible(False)
        
        if results is not None and not results.empty:
            # Assess quality of results
            quality_assessor = QualityAssessor()
            quality_report = quality_assessor.assess_batch_quality(results.to_dict('records'))
            
            # Display results in table with quality information
            self.display_standardized_results(results, quality_report)
            
            # Show success message with statistics
            total_records = len(results)
            
            # Check for standardized columns
            lat_col = 'latitude_dd' if 'latitude_dd' in results.columns else 'latitude'
            lon_col = 'longitude_dd' if 'longitude_dd' in results.columns else 'longitude'
            
            records_with_coords = len(results[results[lat_col].notna() & results[lon_col].notna()])
            
            # Quality summary
            quality_dist = quality_report['quality_distribution']
            excellent_count = quality_dist.get('Excellent', 0)
            good_count = quality_dist.get('Good', 0)
            
            QMessageBox.information(
                self, 
                "Extracción Completada", 
                f"Se extrajeron {total_records} direcciones del documento.\n"
                f"De ellas, {records_with_coords} tienen coordenadas geográficas.\n\n"
                f"Calidad: {excellent_count} excelentes, {good_count} buenas\n"
                f"Calidad general: {quality_report['overall_quality']}\n\n"
                f"Los resultados se muestran en la tabla de abajo."
            )
        else:
            QMessageBox.warning(
                self, 
                "Sin Resultados", 
                "No se encontraron direcciones en el documento.\n"
                "Intenta con otro documento o verifica la configuración."
            )
    
    def display_standardized_results(self, results, quality_report):
        """Display standardized results with quality assessment"""
        # Clear existing data
        self.results_table.setRowCount(0)
        
        if results.empty:
            return
        
        # Map columns to standardized format
        original_col = 'original_text' if 'original_text' in results.columns else 'text'
        address_col = 'inferred_address' if 'inferred_address' in results.columns else 'normalized_address'
        confidence_col = 'confidence_level' if 'confidence_level' in results.columns else 'confidence'
        lat_col = 'latitude_dd' if 'latitude_dd' in results.columns else 'latitude'
        lon_col = 'longitude_dd' if 'longitude_dd' in results.columns else 'longitude'
        
        # Set table row count
        self.results_table.setRowCount(len(results))
        
        # Get individual quality assessments
        quality_assessor = QualityAssessor()
        
        for i, (_, row) in enumerate(results.iterrows()):
            # Original text
            original_text = str(row.get(original_col, ''))[:100] + "..." if len(str(row.get(original_col, ''))) > 100 else str(row.get(original_col, ''))
            self.results_table.setItem(i, 0, QTableWidgetItem(original_text))
            
            # Inferred address
            address = str(row.get(address_col, ''))
            self.results_table.setItem(i, 1, QTableWidgetItem(address))
            
            # Confidence
            confidence = row.get(confidence_col, 0.0)
            confidence_item = QTableWidgetItem(f"{confidence:.2f}")
            # Color code confidence
            if confidence >= 0.8:
                confidence_item.setStyleSheet("background-color: #d4edda; color: #155724;")
            elif confidence >= 0.5:
                confidence_item.setStyleSheet("background-color: #fff3cd; color: #856404;")
            else:
                confidence_item.setStyleSheet("background-color: #f8d7da; color: #721c24;")
            self.results_table.setItem(i, 2, confidence_item)
            
            # Coordinates
            lat = row.get(lat_col, 0.0)
            lon = row.get(lon_col, 0.0)
            self.results_table.setItem(i, 3, QTableWidgetItem(f"{lat:.6f}"))
            self.results_table.setItem(i, 4, QTableWidgetItem(f"{lon:.6f}"))
            
            # Quality assessment
            row_dict = row.to_dict()
            quality_assessment = quality_assessor.assess_extraction_quality(row_dict)
            
            if hasattr(quality_assessment, 'quality_label'):
                quality_label = quality_assessment.quality_label
                quality_score = quality_assessment.quality_score
            else:
                quality_label = quality_assessment['quality_label']
                quality_score = quality_assessment['quality_score']
            
            quality_item = QTableWidgetItem(f"{quality_label} ({quality_score:.2f})")
            # Color code quality
            if quality_label == 'Excellent':
                quality_item.setStyleSheet("background-color: #d1ecf1; color: #0c5460;")
            elif quality_label == 'Good':
                quality_item.setStyleSheet("background-color: #d4edda; color: #155724;")
            elif quality_label == 'Fair':
                quality_item.setStyleSheet("background-color: #fff3cd; color: #856404;")
            else:
                quality_item.setStyleSheet("background-color: #f8d7da; color: #721c24;")
            self.results_table.setItem(i, 5, quality_item)
        
        # Update summary
        self.update_summary_display(results, quality_report)
        
        # Store current results for export
        self.current_results = results
        
        # Enable export buttons
        self.export_csv_btn.setEnabled(True)
        self.export_excel_btn.setEnabled(True)
        self.export_shapefile_btn.setEnabled(True)
        self.export_arcgis_btn.setEnabled(True)
    
    def update_summary_display(self, results, quality_report):
        """Update summary and quality displays"""
        if results.empty:
            self.summary_label.setText("No data processed yet")
            self.quality_label.setText("No quality assessment available")
            return
        
        # Summary statistics
        total_records = len(results)
        lat_col = 'latitude_dd' if 'latitude_dd' in results.columns else 'latitude'
        lon_col = 'longitude_dd' if 'longitude_dd' in results.columns else 'longitude'
        records_with_coords = len(results[results[lat_col].notna() & results[lon_col].notna()])
        
        confidence_col = 'confidence_level' if 'confidence_level' in results.columns else 'confidence'
        avg_confidence = results[confidence_col].mean() if confidence_col in results.columns else 0.0
        
        summary_text = (
            f"Total Records: {total_records} | "
            f"With Coordinates: {records_with_coords} | "
            f"Average Confidence: {avg_confidence:.2f}"
        )
        self.summary_label.setText(summary_text)
        
        # Quality assessment display
        quality_dist = quality_report['quality_distribution']
        overall_quality = quality_report['overall_quality']
        avg_score = quality_report.get('average_score', 0.0)
        
        quality_text = (
            f"Overall Quality: {overall_quality} (Score: {avg_score:.2f})\n"
            f"Distribution: {quality_dist['Excellent']} Excellent, {quality_dist['Good']} Good, "
            f"{quality_dist['Fair']} Fair, {quality_dist['Poor']} Poor"
        )
        
        # Color code overall quality
        if overall_quality == 'Excellent':
            color = "#d1ecf1"
        elif overall_quality == 'Good':
            color = "#d4edda"
        elif overall_quality == 'Fair':
            color = "#fff3cd"
        else:
            color = "#f8d7da"
        
        self.quality_label.setText(quality_text)
        self.quality_label.setStyleSheet(f"padding: 10px; background: {color}; border-radius: 5px; font-size: 12px;")
        
        # Quality statistics
        recommendations = quality_report.get('recommendations', [])
        if recommendations:
            rec_text = "Recommendations:\n• " + "\n• ".join(recommendations[:3])
            self.quality_stats_label.setText(rec_text)
        else:
            self.quality_stats_label.setText("No specific recommendations")
    
    def display_results(self, results):
        """Display results in the table with better formatting"""
        # Clear existing data
        self.results_table.setRowCount(0)
        
        # Set table headers
        headers = ['#', 'Texto Original', 'Dirección Normalizada', 'Latitud', 'Longitud', 'X', 'Y']
        self.results_table.setColumnCount(len(headers))
        self.results_table.setHorizontalHeaderLabels(headers)
        
        # Populate table
        for i, (index, row) in enumerate(results.iterrows()):
            self.results_table.insertRow(i)
            
            # Row number
            self.results_table.setItem(i, 0, QTableWidgetItem(str(i + 1)))
            
            # Original text (truncated if too long)
            original_text = str(row['original_text'])
            if len(original_text) > 50:
                original_text = original_text[:47] + "..."
            self.results_table.setItem(i, 1, QTableWidgetItem(original_text))
            
            # Normalized address
            normalized = str(row['normalized_address'])
            if len(normalized) > 50:
                normalized = normalized[:47] + "..."
            self.results_table.setItem(i, 2, QTableWidgetItem(normalized))
            
            # Coordinates with formatting
            lat = row['latitude']
            lon = row['longitude']
            x = row['x']
            y = row['y']
            
            self.results_table.setItem(i, 3, QTableWidgetItem(f"{lat:.6f}" if pd.notna(lat) else "N/A"))
            self.results_table.setItem(i, 4, QTableWidgetItem(f"{lon:.6f}" if pd.notna(lon) else "N/A"))
            self.results_table.setItem(i, 5, QTableWidgetItem(f"{x:.6f}" if pd.notna(x) else "N/A"))
            self.results_table.setItem(i, 6, QTableWidgetItem(f"{y:.6f}" if pd.notna(y) else "N/A"))
        
        # Auto-resize columns
        self.results_table.resizeColumnsToContents()
        
        # Show summary in status
        total_records = len(results)
        records_with_coords = len(results[results['latitude'].notna() & results['longitude'].notna()])
        self.log_status(f"Extraction completed: {total_records} addresses, {records_with_coords} with coordinates")
        
        # Store results for export
        self.current_results = results
        
        # Enable export buttons
        self.enable_export_buttons()
        
        # Enable map controls and refresh map automatically
        if hasattr(self, 'refresh_map_btn'):
            self.refresh_map_btn.setEnabled(True)
            self.center_map_btn.setEnabled(True)
            # Auto-refresh the map with new data
            self.refresh_map()
    
    def setup_connections(self):
        """Setup signal connections"""
        # This method is called in __init__ but doesn't need to do anything
        # as connections are already set up in create_main_tab and other methods
        pass
    
    def processing_error(self, error_message):
        """Handle processing errors"""
        # Re-enable buttons
        self.process_btn.setEnabled(True)
        self.upload_btn.setEnabled(True)
        self.progress_bar.setVisible(False)
        
        QMessageBox.critical(self, "Error", f"Processing error: {error_message}")
        self.log_status(f"ERROR: {error_message}")
    
    def enable_export_buttons(self):
        """Enable export buttons after processing"""
        self.export_csv_btn.setEnabled(True)
        self.export_excel_btn.setEnabled(True)
        self.export_shapefile_btn.setEnabled(True)
        self.export_arcgis_btn.setEnabled(True)
    
    def export_data(self, export_type):
        """Export data in various formats"""
        if self.current_results is None:
            QMessageBox.warning(self, "Warning", "No data to export. Please process a document first.")
            return
        
        try:
            # Debug logging
            self.log_status(f"Starting export to {export_type}")
            self.log_status(f"Data shape: {self.current_results.shape}")
            self.log_status(f"Data columns: {list(self.current_results.columns)}")
            
            exporter = DataExporter()
            
            # Validate data before export
            validation = exporter.validate_export_data(self.current_results)
            if not validation['valid']:
                QMessageBox.warning(self, "Validation Warning", 
                                  f"Data validation failed: {validation['errors']}")
                return
            
            self.log_status(f"Data validation passed: {validation['status']}")
            
            success = False
            
            if export_type == 'csv':
                file_path, _ = QFileDialog.getSaveFileName(
                    self, "Save CSV", "", "CSV Files (*.csv)")
                if file_path:
                    self.log_status(f"Exporting CSV to: {file_path}")
                    success = exporter.export_to_csv(self.current_results, file_path)
                    
            elif export_type == 'excel':
                file_path, _ = QFileDialog.getSaveFileName(
                    self, "Save Excel", "", "Excel Files (*.xlsx)")
                if file_path:
                    self.log_status(f"Exporting Excel to: {file_path}")
                    success = exporter.export_to_excel(self.current_results, file_path)
                    
            elif export_type == 'shapefile':
                file_path = QFileDialog.getExistingDirectory(
                    self, "Select Directory for Shapefile")
                if file_path:
                    self.log_status(f"Exporting Shapefile to: {file_path}")
                    success = exporter.export_to_shapefile(self.current_results, file_path)
                    
            elif export_type == 'arcgis':
                file_path = QFileDialog.getExistingDirectory(
                    self, "Select Directory for ArcGIS Feature Class")
                if file_path:
                    self.log_status(f"Exporting ArcGIS to: {file_path}")
                    success = exporter.export_to_arcgis(self.current_results, file_path)
            
            if success:
                QMessageBox.information(self, "Success", f"Data exported successfully to {export_type.upper()} format!")
                self.log_status(f"Export completed: {export_type.upper()}")
            else:
                QMessageBox.warning(self, "Export Warning", f"Export to {export_type.upper()} completed with warnings. Check logs for details.")
            
        except Exception as e:
            error_msg = f"Failed to export data: {str(e)}"
            QMessageBox.critical(self, "Export Error", error_msg)
            self.log_status(f"Export ERROR: {error_msg}")
            
            # Additional debug info
            import traceback
            self.log_status(f"Export ERROR traceback: {traceback.format_exc()}")
    
    def create_map_tab(self):
        """Create the map visualization tab"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # Map controls section
        controls_group = QGroupBox("Map Controls")
        controls_layout = QHBoxLayout(controls_group)
        
        # Refresh map button
        self.refresh_map_btn = QPushButton("Refresh Map")
        self.refresh_map_btn.clicked.connect(self.refresh_map)
        self.refresh_map_btn.setEnabled(False)
        controls_layout.addWidget(self.refresh_map_btn)
        
        # Center map button
        self.center_map_btn = QPushButton("Center on Data")
        self.center_map_btn.clicked.connect(self.center_map_on_data)
        self.center_map_btn.setEnabled(False)
        controls_layout.addWidget(self.center_map_btn)
        
        # Map type selector
        controls_layout.addWidget(QLabel("Map Type:"))
        self.map_type_combo = QComboBox()
        self.map_type_combo.addItems(["OpenStreetMap", "Satellite", "Terrain"])
        self.map_type_combo.currentTextChanged.connect(self.change_map_type)
        controls_layout.addWidget(self.map_type_combo)
        
        controls_layout.addStretch()
        layout.addWidget(controls_group)
        
        # Map display section
        map_group = QGroupBox("Geographic Visualization")
        map_layout = QVBoxLayout(map_group)
        
        # Create HTML content for offline map
        self.map_html_content = self.create_offline_map_html()
        
        # Create real map display using QWebEngineView
        try:
            from PyQt5.QtWebEngineWidgets import QWebEngineView
            self.map_view = QWebEngineView()
            self.map_view.setMinimumHeight(500)
            map_layout.addWidget(self.map_view)
            self.web_engine_available = True
            
            # Load initial map
            self.load_map()
            
        except ImportError:
            # Fallback: show message if WebEngine not available
            self.map_view = QLabel("Map visualization requires PyQt5.QtWebEngineWidgets\n"
                                  "Install with: pip install PyQtWebEngine")
            self.map_view.setAlignment(Qt.AlignCenter)
            self.map_view.setStyleSheet("padding: 20px; background: #f0f0f0; border: 1px solid #ccc;")
            map_layout.addWidget(self.map_view)
            self.web_engine_available = False
        
        layout.addWidget(map_group)
        
        # Map info section
        info_group = QGroupBox("Map Information")
        info_layout = QVBoxLayout(info_group)
        
        self.map_info_label = QLabel("No data loaded. Process a document to view locations on the map.")
        self.map_info_label.setStyleSheet("padding: 10px; background: #f0f0f0; border-radius: 5px; color: #333; font-size: 13px;")
        info_layout.addWidget(self.map_info_label)
        
        layout.addWidget(info_group)
        
        return tab
    
    def create_offline_map_html(self):
        """Create completely offline map HTML content"""
        html_content = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>GIS Address Map - Offline</title>
            <meta charset="utf-8">
            <style>
                body { margin: 0; padding: 0; font-family: Arial, sans-serif; }
                #map { width: 100%; height: 500px; background: #e8f4f8; position: relative; }
                .map-controls { position: absolute; top: 10px; right: 10px; z-index: 1000; }
                .map-controls button { margin: 2px; padding: 8px 12px; border: none; border-radius: 4px; background: white; cursor: pointer; box-shadow: 0 2px 4px rgba(0,0,0,0.2); }
                .map-controls button:hover { background: #f0f0f0; }
                .coordinate-grid { position: absolute; bottom: 10px; left: 10px; background: rgba(255,255,255,0.9); padding: 5px; border-radius: 4px; font-size: 12px; }
                .point-info { position: absolute; top: 50px; left: 10px; background: rgba(255,255,255,0.9); padding: 10px; border-radius: 4px; max-width: 300px; display: none; }
                .point-info h4 { margin: 0 0 5px 0; color: #333; }
                .point-info p { margin: 0; color: #666; }
            </style>
        </head>
        <body>
            <div id="map">
                <div class="map-controls">
                    <button onclick="zoomIn()" title="Zoom In">+</button>
                    <button onclick="zoomOut()" title="Zoom Out">-</button>
                    <button onclick="resetView()" title="Reset View">⌂</button>
                    <button onclick="toggleGrid()" title="Toggle Grid">⊞</button>
                </div>
                <div class="coordinate-grid" id="coordinateGrid">Click on map to see coordinates</div>
                <div class="point-info" id="pointInfo"></div>
            </div>
            
            <script>
                // Simple offline map implementation
                var canvas = document.createElement('canvas');
                var ctx = canvas.getContext('2d');
                var mapDiv = document.getElementById('map');
                mapDiv.appendChild(canvas);
                
                // Map state
                var mapState = {
                    centerX: 0,
                    centerY: 0,
                    zoom: 2,
                    points: [],
                    showGrid: true,
                    canvas: canvas,
                    ctx: ctx
                };
                
                // Initialize canvas
                function initCanvas() {
                    canvas.width = mapDiv.offsetWidth;
                    canvas.height = mapDiv.offsetHeight;
                    drawMap();
                }
                
                // Draw the map
                function drawMap() {
                    var width = canvas.width;
                    var height = canvas.height;
                    
                    // Clear canvas
                    ctx.clearRect(0, 0, width, height);
                    
                    // Draw background
                    ctx.fillStyle = '#e8f4f8';
                    ctx.fillRect(0, 0, width, height);
                    
                    // Draw grid if enabled
                    if (mapState.showGrid) {
                        drawGrid(width, height);
                    }
                    
                    // Draw points
                    drawPoints();
                    
                    // Draw center marker
                    drawCenterMarker(width, height);
                }
                
                // Draw coordinate grid
                function drawGrid(width, height) {
                    ctx.strokeStyle = '#d0d0d0';
                    ctx.lineWidth = 1;
                    
                    var gridSize = 50 * Math.pow(2, mapState.zoom);
                    var startX = (mapState.centerX % gridSize) - gridSize;
                    var startY = (mapState.centerY % gridSize) - gridSize;
                    
                    for (var x = startX; x < width + gridSize; x += gridSize) {
                        ctx.beginPath();
                        ctx.moveTo(x, 0);
                        ctx.lineTo(x, height);
                        ctx.stroke();
                    }
                    
                    for (var y = startY; y < height + gridSize; y += gridSize) {
                        ctx.beginPath();
                        ctx.moveTo(0, y);
                        ctx.lineTo(width, y);
                        ctx.stroke();
                    }
                }
                
                // Draw center marker
                function drawCenterMarker(width, height) {
                    var centerX = width / 2;
                    var centerY = height / 2;
                    
                    ctx.strokeStyle = '#0066cc';
                    ctx.lineWidth = 2;
                    ctx.setLineDash([5, 5]);
                    
                    ctx.beginPath();
                    ctx.moveTo(centerX - 10, centerY);
                    ctx.lineTo(centerX + 10, centerY);
                    ctx.moveTo(centerX, centerY - 10);
                    ctx.lineTo(centerX, centerY + 10);
                    ctx.stroke();
                    
                    ctx.setLineDash([]);
                }
                
                // Draw points
                function drawPoints() {
                    mapState.points.forEach(function(point) {
                        var screenX = (point.lon - mapState.centerX) * Math.pow(2, mapState.zoom) + canvas.width / 2;
                        var screenY = (point.lat - mapState.centerY) * Math.pow(2, mapState.zoom) + canvas.height / 2;
                        
                        if (screenX >= 0 && screenX < canvas.width && screenY >= 0 && screenY < canvas.height) {
                            // Draw point
                            ctx.fillStyle = '#ff4444';
                            ctx.beginPath();
                            ctx.arc(screenX, screenY, 6, 0, 2 * Math.PI);
                            ctx.fill();
                            
                            // Draw border
                            ctx.strokeStyle = '#ffffff';
                            ctx.lineWidth = 2;
                            ctx.stroke();
                            
                            // Draw label
                            ctx.fillStyle = '#333333';
                            ctx.font = '12px Arial';
                            ctx.textAlign = 'center';
                            ctx.fillText(point.address.substring(0, 20), screenX, screenY - 10);
                        }
                    });
                }
                
                // Mouse events
                canvas.addEventListener('click', function(e) {
                    var rect = canvas.getBoundingClientRect();
                    var x = e.clientX - rect.left;
                    var y = e.clientY - rect.top;
                    
                    // Convert to coordinates
                    var lon = (x - canvas.width / 2) / Math.pow(2, mapState.zoom) + mapState.centerX;
                    var lat = (y - canvas.height / 2) / Math.pow(2, mapState.zoom) + mapState.centerY;
                    
                    // Update coordinate display
                    document.getElementById('coordinateGrid').innerHTML = 
                        'Lon: ' + lon.toFixed(6) + '<br>Lat: ' + lat.toFixed(6);
                    
                    // Check if clicked on a point
                    var clickedPoint = findPointAt(x, y);
                    if (clickedPoint) {
                        showPointInfo(clickedPoint, x, y);
                    } else {
                        hidePointInfo();
                    }
                });
                
                // Mouse wheel for zoom
                canvas.addEventListener('wheel', function(e) {
                    e.preventDefault();
                    var zoomChange = e.deltaY > 0 ? -1 : 1;
                    mapState.zoom = Math.max(1, Math.min(10, mapState.zoom + zoomChange * 0.5));
                    drawMap();
                });
                
                // Mouse drag for pan
                var isDragging = false;
                var lastX, lastY;
                
                canvas.addEventListener('mousedown', function(e) {
                    isDragging = true;
                    lastX = e.clientX;
                    lastY = e.clientY;
                    canvas.style.cursor = 'grabbing';
                });
                
                canvas.addEventListener('mousemove', function(e) {
                    if (isDragging) {
                        var deltaX = e.clientX - lastX;
                        var deltaY = e.clientY - lastY;
                        
                        mapState.centerX -= deltaX / Math.pow(2, mapState.zoom);
                        mapState.centerY -= deltaY / Math.pow(2, mapState.zoom);
                        
                        lastX = e.clientX;
                        lastY = e.clientY;
                        
                        drawMap();
                    }
                });
                
                canvas.addEventListener('mouseup', function() {
                    isDragging = false;
                    canvas.style.cursor = 'grab';
                });
                
                // Find point at screen coordinates
                function findPointAt(screenX, screenY) {
                    for (var i = mapState.points.length - 1; i >= 0; i--) {
                        var point = mapState.points[i];
                        var pointScreenX = (point.lon - mapState.centerX) * Math.pow(2, mapState.zoom) + canvas.width / 2;
                        var pointScreenY = (point.lat - mapState.centerY) * Math.pow(2, mapState.zoom) + canvas.height / 2;
                        
                        var distance = Math.sqrt(Math.pow(screenX - pointScreenX, 2) + Math.pow(screenY - pointScreenY, 2));
                        if (distance <= 10) {
                            return point;
                        }
                    }
                    return null;
                }
                
                // Show point information
                function showPointInfo(point, screenX, screenY) {
                    var infoDiv = document.getElementById('pointInfo');
                    infoDiv.innerHTML = '<h4>Location</h4><p>' + point.address + '</p><p>Lon: ' + point.lon.toFixed(6) + '</p><p>Lat: ' + point.lat.toFixed(6) + '</p>';
                    infoDiv.style.display = 'block';
                    infoDiv.style.left = (screenX + 10) + 'px';
                    infoDiv.style.top = (screenY - 50) + 'px';
                }
                
                // Hide point information
                function hidePointInfo() {
                    document.getElementById('pointInfo').style.display = 'none';
                }
                
                // Control functions
                function zoomIn() { 
                    mapState.zoom = Math.min(10, mapState.zoom + 1); 
                    drawMap(); 
                }
                
                function zoomOut() { 
                    mapState.zoom = Math.max(1, mapState.zoom - 1); 
                    drawMap(); 
                }
                
                function resetView() { 
                    mapState.centerX = 0; 
                    mapState.centerY = 0; 
                    mapState.zoom = 2; 
                    drawMap(); 
                }
                
                function toggleGrid() { 
                    mapState.showGrid = !mapState.showGrid; 
                    drawMap(); 
                }
                
                // Functions called from Python
                function addPoint(lon, lat, address) {
                    mapState.points.push({lon: lon, lat: lat, address: address});
                    drawMap();
                }
                
                function clearPoints() {
                    mapState.points = [];
                    drawMap();
                }
                
                function fitToPoints() {
                    if (mapState.points.length === 0) return;
                    
                    var minLon = Math.min(...mapState.points.map(p => p.lon));
                    var maxLon = Math.max(...mapState.points.map(p => p.lon));
                    var minLat = Math.min(...mapState.points.map(p => p.lat));
                    var maxLat = Math.max(...mapState.points.map(p => p.lat));
                    
                    mapState.centerX = (minLon + maxLon) / 2;
                    mapState.centerY = (minLat + maxLat) / 2;
                    
                    var lonRange = maxLon - minLon;
                    var latRange = maxLat - minLat;
                    var range = Math.max(lonRange, latRange);
                    
                    mapState.zoom = Math.max(1, Math.min(10, 10 - Math.log2(range)));
                    
                    drawMap();
                }
                
                // Initialize
                initCanvas();
                window.addEventListener('resize', initCanvas);
            </script>
        </body>
        </html>
        """
        return html_content
    
    def load_map(self):
        """Load the initial map"""
        if not self.web_engine_available:
            return
        
        # Create HTML content for the map
        html_content = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>GIS Address Map</title>
            <meta charset="utf-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" />
            <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
            <style>
                body { margin: 0; padding: 0; }
                #map { width: 100%; height: 500px; }
                .map-info { 
                    position: absolute; 
                    top: 10px; 
                    left: 10px; 
                    background: rgba(255, 255, 255, 0.95); 
                    padding: 12px; 
                    border-radius: 6px; 
                    box-shadow: 0 3px 10px rgba(0,0,0,0.15);
                    z-index: 1000;
                    max-width: 280px;
                    border: 1px solid #0066cc;
                }
                .map-info h3 { 
                    margin: 0 0 8px 0; 
                    color: #0066cc; 
                    font-size: 14px;
                    font-weight: bold;
                }
                .map-info p { 
                    margin: 4px 0; 
                    color: #333; 
                    font-size: 12px;
                    line-height: 1.3;
                }
            </style>
        </head>
        <body>
            <div id="map"></div>
            <div class="map-info">
                <h3>🗺️ GIS Address Map</h3>
                <p>Process a document to see locations marked on the map.</p>
                <p>Click on markers to see address details.</p>
            </div>
            
            <script>
                // Initialize map
                var map = L.map('map').setView([0, 0], 2);
                
                // Add OpenStreetMap tiles
                L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
                    attribution: '© <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
                }).addTo(map);
                
                // Store markers for later manipulation
                var markers = [];
                
                // Function to add points (called from Python)
                function addPoint(lon, lat, address) {
                    var marker = L.marker([lat, lon]).addTo(map);
                    
                    // Create popup content
                    var popupContent = '<div style="min-width: 200px;">' +
                                     '<h4 style="margin: 0 0 10px 0; color: #333;">📍 Location</h4>' +
                                     '<p style="margin: 5px 0; color: #666;"><strong>Address:</strong><br>' + address + '</p>' +
                                     '<p style="margin: 5px 0; color: #666;"><strong>Coordinates:</strong><br>' +
                                     'Lon: ' + lon.toFixed(6) + '<br>' +
                                     'Lat: ' + lat.toFixed(6) + '</p>' +
                                     '</div>';
                    
                    marker.bindPopup(popupContent);
                    markers.push(marker);
                    
                    return marker;
                }
                
                // Function to clear all points
                function clearPoints() {
                    markers.forEach(function(marker) {
                        map.removeLayer(marker);
                    });
                    markers = [];
                }
                
                // Function to fit view to all points
                function fitToPoints() {
                    if (markers.length > 0) {
                        var group = new L.featureGroup(markers);
                        map.fitBounds(group.getBounds().pad(0.1));
                    }
                }
                
                // Function to change map style
                function changeMapStyle(style) {
                    map.removeLayer(map._layers[Object.keys(map._layers)[0]]);
                    
                    if (style === 'satellite') {
                        L.tileLayer('https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}', {
                            attribution: '© Esri'
                        }).addTo(map);
                    } else if (style === 'terrain') {
                        L.tileLayer('https://server.arcgisonline.com/ArcGIS/rest/services/World_Topo_Map/MapServer/tile/{z}/{y}/{x}', {
                            attribution: '© Esri'
                        }).addTo(map);
                    } else {
                        L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
                            attribution: '© <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
                        }).addTo(map);
                    }
                }
                
                // Make functions globally available
                window.addPoint = addPoint;
                window.clearPoints = clearPoints;
                window.fitToPoints = fitToPoints;
                window.changeMapStyle = changeMapStyle;
            </script>
        </body>
        </html>
        """
        
        self.map_view.setHtml(html_content)
    
    def refresh_map(self):
        """Refresh the map with current data"""
        if self.current_results is None or not self.web_engine_available:
            return
        
        try:
            # Count points with coordinates
            total_points = len(self.current_results[self.current_results['longitude'].notna() & self.current_results['latitude'].notna()])
            
            if total_points > 0:
                # Clear existing points
                self.map_view.page().runJavaScript("clearPoints();")
                
                # Add new points from current results
                for _, row in self.current_results.iterrows():
                    if pd.notna(row['longitude']) and pd.notna(row['latitude']):
                        lon = float(row['longitude'])
                        lat = float(row['latitude'])
                        address = str(row['normalized_address']).strip()
                        
                        # Escape quotes in address for JavaScript
                        address = address.replace("'", "\\'").replace('"', '\\"')
                        
                        # Add point to map
                        js_code = f"addPoint({lon}, {lat}, '{address}');"
                        self.map_view.page().runJavaScript(js_code)
                
                # Fit map to show all points
                self.map_view.page().runJavaScript("fitToPoints();")
                
                # Update map info
                self.map_info_label.setText(f"Map loaded with {total_points} locations")
            else:
                self.map_info_label.setText("No geographic data available")
            
            # Enable map controls
            self.refresh_map_btn.setEnabled(True)
            self.center_map_btn.setEnabled(True)
            
        except Exception as e:
            self.log_status(f"Map refresh error: {str(e)}")
            self.map_info_label.setText(f"Map error: {str(e)}")
    
    def center_map_on_data(self):
        """Center the map on the loaded data"""
        if not self.web_engine_available:
            return
        
        try:
            self.map_view.page().runJavaScript("fitToPoints();")
            self.log_status("Map centered on data")
        except Exception as e:
            self.log_status(f"Map centering error: {str(e)}")
    
    def change_map_type(self, map_type):
        """Change the map type"""
        if not self.web_engine_available:
            return
        
        try:
            if map_type == "Satellite":
                self.map_view.page().runJavaScript("changeMapStyle('satellite');")
            elif map_type == "Terrain":
                self.map_view.page().runJavaScript("changeMapStyle('terrain');")
            else:  # OpenStreetMap
                self.map_view.page().runJavaScript("changeMapStyle('osm');")
            
            self.log_status(f"Map type changed to: {map_type}")
        except Exception as e:
            self.log_status(f"Map type change error: {str(e)}")


def main():
    """Main application entry point"""
    # Set OpenGL context sharing for WebEngine
    QCoreApplication.setAttribute(Qt.AA_ShareOpenGLContexts, True)
    
    app = QApplication(sys.argv)
    
    # Set application style
    app.setStyle('Fusion')
    
    # Create and show main window
    window = MainWindow()
    window.show()
    
    # Start event loop
    sys.exit(app.exec_())


if __name__ == "__main__":
    main() 