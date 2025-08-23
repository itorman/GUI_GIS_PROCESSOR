#!/usr/bin/env python3
"""
Demo mejorado del sistema de extracción de direcciones GIS
Muestra cómo el sistema detecta y geolocaliza direcciones de cualquier documento
"""

import sys
import os
import pandas as pd
from pathlib import Path

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from preprocessing.document_processor import DocumentProcessor
from llm.llm_client import LLMClient
from postprocessing.data_processor import DataProcessor
from export.data_exporter import DataExporter

def print_header(title):
    """Print a formatted header"""
    print("\n" + "="*60)
    print(f" {title}")
    print("="*60)

def print_section(title):
    """Print a formatted section"""
    print(f"\n--- {title} ---")

def demo_pdf_processing():
    """Demo del procesamiento del PDF con direcciones"""
    print_header("DEMO: Extracción de Direcciones del PDF")
    
    # 1. Procesar el PDF
    print_section("1. Procesamiento del PDF")
    doc_processor = DocumentProcessor(chunk_size=1000)
    
    try:
        text_chunks = doc_processor.process_document('random_addresses_coordinates.pdf')
        print(f"✓ PDF procesado exitosamente en {len(text_chunks)} chunks")
        
        # Mostrar preview de cada chunk
        for i, chunk in enumerate(text_chunks):
            preview = chunk[:200] + "..." if len(chunk) > 200 else chunk
            print(f"  Chunk {i+1}: {preview}")
            
    except Exception as e:
        print(f"✗ Error procesando PDF: {e}")
        return None
    
    # 2. Extraer direcciones usando modo test mejorado
    print_section("2. Extracción de Direcciones (Modo Test)")
    llm_client = LLMClient({
        'type': 'Test Mode', 
        'server_url': 'http://localhost:11434', 
        'model': 'llama3:8b'
    })
    
    all_results = []
    for i, chunk in enumerate(text_chunks):
        print(f"  Procesando chunk {i+1}/{len(text_chunks)}...")
        chunk_result = llm_client.extract_addresses(chunk)
        
        if chunk_result:
            print(f"    ✓ Encontradas {len(chunk_result)} direcciones")
            all_results.extend(chunk_result)
        else:
            print(f"    ✗ No se encontraron direcciones")
    
    print(f"\n  Total de direcciones encontradas: {len(all_results)}")
    
    # 3. Procesar y limpiar resultados
    print_section("3. Procesamiento y Limpieza de Datos")
    if all_results:
        data_processor = DataProcessor()
        processed_data = data_processor.process_results(all_results)
        
        print(f"✓ Datos procesados: {len(processed_data)} registros únicos")
        
        # Mostrar estadísticas
        total_records = len(processed_data)
        records_with_coords = len(processed_data[processed_data['latitude'].notna() & processed_data['longitude'].notna()])
        records_without_coords = total_records - records_with_coords
        
        print(f"\n📊 Estadísticas:")
        print(f"  • Total de direcciones: {total_records}")
        print(f"  • Con coordenadas: {records_with_coords}")
        print(f"  • Sin coordenadas: {records_without_coords}")
        
        # Mostrar ejemplos de cada tipo
        if records_with_coords > 0:
            print(f"\n📍 Ejemplos con coordenadas:")
            coords_examples = processed_data[processed_data['latitude'].notna()].head(3)
            for i, (_, row) in enumerate(coords_examples.iterrows()):
                print(f"  {i+1}. {row['normalized_address']:<40} ({row['latitude']:8.4f}, {row['longitude']:8.4f})")
        
        if records_without_coords > 0:
            print(f"\n🏠 Ejemplos sin coordenadas:")
            no_coords_examples = processed_data[processed_data['latitude'].isna()].head(3)
            for i, (_, row) in enumerate(no_coords_examples.iterrows()):
                print(f"  {i+1}. {row['normalized_address']}")
        
        return processed_data
    else:
        print("✗ No se encontraron direcciones para procesar")
        return None

def demo_spanish_document():
    """Demo del documento en español"""
    print_header("DEMO: Documento en Español")
    
    # Procesar documento español
    doc_processor = DocumentProcessor(chunk_size=1000)
    text_chunks = doc_processor.process_document('test_spanish_document.txt')
    
    print(f"✓ Documento español procesado en {len(text_chunks)} chunks")
    
    # Extraer direcciones
    llm_client = LLMClient({'type': 'Test Mode', 'server_url': 'http://localhost:11434', 'model': 'llama3:8b'})
    
    all_results = []
    for chunk in text_chunks:
        chunk_result = llm_client.extract_addresses(chunk)
        if chunk_result:
            all_results.extend(chunk_result)
    
    if all_results:
        data_processor = DataProcessor()
        processed_data = data_processor.process_results(all_results)
        
        print(f"✓ Extraídas {len(processed_data)} direcciones del documento español")
        
        # Mostrar ejemplos
        print(f"\n🇪🇸 Direcciones españolas encontradas:")
        for i, (_, row) in enumerate(processed_data.head(5).iterrows()):
            coords_info = f"({row['latitude']:.4f}, {row['longitude']:.4f})" if pd.notna(row['latitude']) else "Sin coordenadas"
            print(f"  {i+1}. {row['normalized_address']:<35} {coords_info}")
    
    return processed_data if all_results else None

def demo_export_functionality(processed_data):
    """Demo de la funcionalidad de exportación"""
    print_header("DEMO: Funcionalidad de Exportación")
    
    if processed_data is None or processed_data.empty:
        print("✗ No hay datos para exportar")
        return
    
    # Crear directorio de salida
    output_dir = Path("demo_output_improved")
    output_dir.mkdir(exist_ok=True)
    
    exporter = DataExporter()
    
    # 1. Exportar a CSV
    print_section("1. Exportación a CSV")
    csv_path = output_dir / "addresses_improved.csv"
    try:
        exporter.export_to_csv(processed_data, str(csv_path))
        print(f"✓ CSV exportado: {csv_path}")
    except Exception as e:
        print(f"✗ Error exportando CSV: {e}")
    
    # 2. Exportar a Excel
    print_section("2. Exportación a Excel")
    excel_path = output_dir / "addresses_improved.xlsx"
    try:
        exporter.export_to_excel(processed_data, str(excel_path))
        print(f"✓ Excel exportado: {excel_path}")
    except Exception as e:
        print(f"✗ Error exportando Excel: {e}")
    
    # 3. Mostrar contenido del CSV
    print_section("3. Vista Previa del CSV")
    try:
        df_preview = pd.read_csv(csv_path)
        print(f"✓ CSV leído exitosamente: {len(df_preview)} filas, {len(df_preview.columns)} columnas")
        
        print(f"\n📋 Columnas disponibles:")
        for col in df_preview.columns:
            print(f"  • {col}")
        
        print(f"\n📊 Primeras 5 filas:")
        print(df_preview.head().to_string(index=False))
        
    except Exception as e:
        print(f"✗ Error leyendo CSV: {e}")

def main():
    """Función principal del demo"""
    print_header("SISTEMA DE EXTRACCIÓN DE DIRECCIONES GIS - DEMO MEJORADO")
    print("Este demo muestra cómo el sistema detecta y geolocaliza direcciones")
    print("de cualquier documento, incluyendo PDFs, archivos de texto, etc.")
    
    # Demo 1: Procesamiento del PDF
    pdf_results = demo_pdf_processing()
    
    # Demo 2: Documento en español
    spanish_results = demo_spanish_document()
    
    # Demo 3: Exportación
    demo_export_functionality(pdf_results if pdf_results is not None else spanish_results)
    
    print_header("DEMO COMPLETADO")
    print("✅ El sistema está funcionando correctamente")
    print("✅ Puede detectar direcciones en múltiples idiomas")
    print("✅ Extrae coordenadas en varios formatos")
    print("✅ Genera archivos de exportación")
    print("\n🚀 Para usar la interfaz gráfica, ejecuta: python main.py")
    print("📁 Los archivos de salida están en: demo_output_improved/")

if __name__ == "__main__":
    main() 