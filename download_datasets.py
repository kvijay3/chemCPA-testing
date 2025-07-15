#!/usr/bin/env python3
"""
ChemCPA Dataset Downloader

This script downloads and prepares datasets for ChemCPA training.
It provides an easy CLI interface to download specific datasets or all datasets.

Usage:
    python download_datasets.py --list                    # List available datasets
    python download_datasets.py --dataset sciplex         # Download Sciplex dataset
    python download_datasets.py --dataset broad           # Download Broad dataset  
    python download_datasets.py --dataset lincs           # Download LINCS dataset
    python download_datasets.py --dataset all             # Download all datasets
    python download_datasets.py --stem-cell-essentials    # Download essential datasets for stem cells
"""

import os
import sys
import argparse
from pathlib import Path
import subprocess
from typing import List, Dict, Optional

# Add raw_data to path
sys.path.append(str(Path(__file__).parent / 'raw_data'))
sys.path.append(str(Path(__file__).parent))

try:
    from raw_data.datasets import (
        DATASETS_INFO, ensure_dataset, list_available_datasets,
        sciplex, norman, lincs_full, adata_biolord_split_30,
        trapnell_final_v7, sciplex_combinatorial, drugbank_all
    )
except ImportError as e:
    print(f"Error: Could not import dataset utilities: {e}")
    print("Make sure you're in the chemCPA directory and raw_data/ exists.")
    print("Trying alternative import...")
    try:
        # Try direct import
        import raw_data.datasets as datasets
        DATASETS_INFO = datasets.DATASETS_INFO
        ensure_dataset = datasets.ensure_dataset
        list_available_datasets = datasets.list_available_datasets
        sciplex = datasets.sciplex
        norman = datasets.norman
        lincs_full = datasets.lincs_full
        adata_biolord_split_30 = datasets.adata_biolord_split_30
        trapnell_final_v7 = datasets.trapnell_final_v7
        sciplex_combinatorial = datasets.sciplex_combinatorial
        drugbank_all = datasets.drugbank_all
    except ImportError as e2:
        print(f"Alternative import also failed: {e2}")
        sys.exit(1)


class ChemCPADatasetDownloader:
    """Main class for downloading ChemCPA datasets"""
    
    def __init__(self):
        self.project_folder = Path("project_folder")
        self.datasets_folder = self.project_folder / "datasets"
        self.embeddings_folder = self.project_folder / "embeddings"
        
        # Create directories, handling existing symlinks
        self._ensure_directory(self.project_folder)
        self._ensure_directory(self.datasets_folder)
        self._ensure_directory(self.embeddings_folder)
    
    def _ensure_directory(self, path: Path):
        """Ensure directory exists, handling symlinks properly"""
        if path.exists():
            if path.is_dir():
                return  # Already exists as directory
            elif path.is_symlink():
                if path.is_dir():
                    return  # Symlink to valid directory
                else:
                    # Broken symlink, remove it
                    path.unlink()
        
        # Create directory
        path.mkdir(parents=True, exist_ok=True)
        
        # Dataset groups for easy downloading
        self.dataset_groups = {
            'stem_cell_essentials': [
                'cpa_binaries',  # Contains sciplex and norman
                'adata_biolord_split_30',  # Broad dataset
                'drugbank_all',  # Drug information
            ],
            'all_datasets': list(DATASETS_INFO.keys()),
            'training_datasets': [
                'cpa_binaries',  # Sciplex + Norman
                'adata_biolord_split_30',  # Broad
                'lincs_full',  # LINCS
            ],
            'embeddings': [
                'rdkit2D_embedding_biolord',
            ]
        }
    
    def print_dataset_info(self):
        """Print detailed information about available datasets"""
        print("\n" + "="*80)
        print("📊 CHEMCPA DATASETS OVERVIEW")
        print("="*80)
        
        dataset_descriptions = {
            'cpa_binaries': {
                'name': 'CPA Binaries (Sciplex + Norman)',
                'size': '~2GB',
                'cells': '~100K cells',
                'drugs': '~200 compounds',
                'description': 'Contains Sciplex and Norman datasets. Good for initial testing.',
                'use_case': 'Training and testing'
            },
            'adata_biolord_split_30': {
                'name': 'Broad Dataset (BioLord)',
                'size': '~5GB',
                'cells': '~1M+ cells',
                'drugs': '~1000+ compounds',
                'description': 'Large-scale dataset from Broad Institute. Best for comprehensive training.',
                'use_case': 'Large-scale training'
            },
            'lincs_full': {
                'name': 'LINCS Dataset',
                'size': '~8GB',
                'cells': '~2M+ cells',
                'drugs': '~2000+ compounds',
                'description': 'Comprehensive drug screening dataset. Very large scale.',
                'use_case': 'Comprehensive drug screening'
            },
            'drugbank_all': {
                'name': 'DrugBank Information',
                'size': '~50MB',
                'cells': 'N/A',
                'drugs': '~13K drugs',
                'description': 'Drug information and annotations.',
                'use_case': 'Drug metadata and analysis'
            },
            'rdkit2D_embedding_biolord': {
                'name': 'RDKit 2D Embeddings',
                'size': '~100MB',
                'cells': 'N/A',
                'drugs': '~1000+ compounds',
                'description': 'Pre-computed chemical embeddings using RDKit.',
                'use_case': 'Chemical representation'
            }
        }
        
        for dataset_key, info in dataset_descriptions.items():
            if dataset_key in DATASETS_INFO:
                print(f"\n🧬 {info['name']}")
                print(f"   • Size: {info['size']}")
                print(f"   • Cells: {info['cells']}")
                print(f"   • Drugs: {info['drugs']}")
                print(f"   • Description: {info['description']}")
                print(f"   • Use case: {info['use_case']}")
                print(f"   • Key: {dataset_key}")
        
        print(f"\n📦 DATASET GROUPS:")
        print(f"   • stem_cell_essentials: Essential datasets for stem cell work")
        print(f"   • training_datasets: Main datasets for model training")
        print(f"   • embeddings: Pre-computed chemical embeddings")
        print(f"   • all_datasets: All available datasets")
        
        print("\n💡 RECOMMENDATIONS:")
        print("   • For quick start: --dataset stem_cell_essentials")
        print("   • For comprehensive training: --dataset training_datasets")
        print("   • For specific dataset: --dataset sciplex, --dataset broad, etc.")
        print("="*80)
    
    def download_dataset_group(self, group_name: str):
        """Download a group of datasets"""
        if group_name not in self.dataset_groups:
            print(f"❌ Error: Dataset group '{group_name}' not found.")
            print(f"Available groups: {list(self.dataset_groups.keys())}")
            return False
        
        datasets = self.dataset_groups[group_name]
        print(f"\n🚀 Downloading dataset group: {group_name}")
        print(f"📦 Datasets to download: {datasets}")
        
        success_count = 0
        for dataset in datasets:
            try:
                print(f"\n{'='*60}")
                print(f"📥 Downloading: {dataset}")
                print(f"{'='*60}")
                
                ensure_dataset(dataset)
                print(f"✅ {dataset} downloaded successfully!")
                success_count += 1
                
            except Exception as e:
                print(f"❌ Error downloading {dataset}: {e}")
                continue
        
        print(f"\n🎉 Download Summary:")
        print(f"   • Successfully downloaded: {success_count}/{len(datasets)} datasets")
        
        if success_count == len(datasets):
            print(f"   • All datasets in '{group_name}' downloaded successfully!")
            return True
        else:
            print(f"   • Some datasets failed to download. Check errors above.")
            return False
    
    def download_single_dataset(self, dataset_name: str):
        """Download a single dataset"""
        # Handle aliases
        dataset_aliases = {
            'sciplex': 'cpa_binaries',
            'norman': 'cpa_binaries', 
            'broad': 'adata_biolord_split_30',
            'biolord': 'adata_biolord_split_30',
            'lincs': 'lincs_full',
            'drugbank': 'drugbank_all',
            'rdkit': 'rdkit2D_embedding_biolord'
        }
        
        actual_dataset = dataset_aliases.get(dataset_name, dataset_name)
        
        if actual_dataset not in DATASETS_INFO:
            print(f"❌ Error: Dataset '{dataset_name}' not found.")
            print(f"Available datasets: {list(DATASETS_INFO.keys())}")
            print(f"Available aliases: {list(dataset_aliases.keys())}")
            return False
        
        try:
            print(f"\n🚀 Downloading: {dataset_name} (actual: {actual_dataset})")
            ensure_dataset(actual_dataset)
            print(f"✅ {dataset_name} downloaded successfully!")
            return True
            
        except Exception as e:
            print(f"❌ Error downloading {dataset_name}: {e}")
            return False
    
    def check_dataset_status(self):
        """Check which datasets are already downloaded"""
        print("\n📋 DATASET STATUS:")
        print("="*60)
        
        for dataset_key, dataset_info in DATASETS_INFO.items():
            relative_path = dataset_info['relative_path']
            full_path = self.project_folder / relative_path
            
            if full_path.exists():
                size = self.get_file_size(full_path)
                print(f"✅ {dataset_key:<30} | {size:>10} | {full_path}")
            else:
                print(f"❌ {dataset_key:<30} | {'Not found':>10} | {full_path}")
        
        print("="*60)
    
    def get_file_size(self, file_path: Path) -> str:
        """Get human-readable file size"""
        try:
            size = file_path.stat().st_size
            for unit in ['B', 'KB', 'MB', 'GB']:
                if size < 1024:
                    return f"{size:.1f} {unit}"
                size /= 1024
            return f"{size:.1f} TB"
        except:
            return "Unknown"
    
    def setup_project_structure(self):
        """Create the necessary project folder structure"""
        print("\n🏗️  Setting up project structure...")
        
        folders_to_create = [
            "project_folder",
            "project_folder/datasets", 
            "project_folder/embeddings",
            "project_folder/embeddings/rdkit",
            "project_folder/embeddings/rdkit/data",
            "project_folder/embeddings/rdkit/data/embeddings",
            "project_folder/embeddings/chemCPA",
            "project_folder/binaries",
            "outputs",
            "logs"
        ]
        
        for folder in folders_to_create:
            Path(folder).mkdir(parents=True, exist_ok=True)
            print(f"   📁 Created: {folder}")
        
        print("✅ Project structure created!")
    
    def install_dependencies(self):
        """Install required dependencies for downloading"""
        print("\n📦 Installing download dependencies...")
        
        dependencies = [
            "gdown",
            "requests", 
            "tqdm"
        ]
        
        for dep in dependencies:
            try:
                subprocess.run([sys.executable, "-m", "pip", "install", dep], 
                             check=True, capture_output=True)
                print(f"   ✅ Installed: {dep}")
            except subprocess.CalledProcessError:
                print(f"   ❌ Failed to install: {dep}")
        
        print("✅ Dependencies installation complete!")


def main():
    parser = argparse.ArgumentParser(
        description='Download datasets for ChemCPA training',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python download_datasets.py --list                    # List all datasets
    python download_datasets.py --info                    # Show detailed dataset info
    python download_datasets.py --status                  # Check download status
    python download_datasets.py --dataset sciplex         # Download Sciplex
    python download_datasets.py --dataset broad           # Download Broad dataset
    python download_datasets.py --dataset lincs           # Download LINCS dataset
    python download_datasets.py --stem-cell-essentials    # Download stem cell essentials
    python download_datasets.py --training-datasets       # Download training datasets
    python download_datasets.py --dataset all             # Download everything
    python download_datasets.py --setup                   # Setup project structure
        """
    )
    
    # Main actions
    parser.add_argument('--list', action='store_true', 
                       help='List all available datasets')
    parser.add_argument('--info', action='store_true',
                       help='Show detailed information about datasets')
    parser.add_argument('--status', action='store_true',
                       help='Check which datasets are already downloaded')
    parser.add_argument('--setup', action='store_true',
                       help='Setup project folder structure')
    
    # Download options
    parser.add_argument('--dataset', type=str,
                       help='Download specific dataset (use "all" for everything)')
    parser.add_argument('--stem-cell-essentials', action='store_true',
                       help='Download essential datasets for stem cell work')
    parser.add_argument('--training-datasets', action='store_true', 
                       help='Download main training datasets')
    parser.add_argument('--embeddings', action='store_true',
                       help='Download pre-computed embeddings')
    
    # Utility options
    parser.add_argument('--install-deps', action='store_true',
                       help='Install required dependencies')
    
    args = parser.parse_args()
    
    # Initialize downloader
    downloader = ChemCPADatasetDownloader()
    
    # Handle arguments
    if args.install_deps:
        downloader.install_dependencies()
        return
    
    if args.setup:
        downloader.setup_project_structure()
        return
    
    if args.list:
        list_available_datasets()
        return
    
    if args.info:
        downloader.print_dataset_info()
        return
    
    if args.status:
        downloader.check_dataset_status()
        return
    
    # Download actions
    if args.stem_cell_essentials:
        downloader.download_dataset_group('stem_cell_essentials')
    elif args.training_datasets:
        downloader.download_dataset_group('training_datasets')
    elif args.embeddings:
        downloader.download_dataset_group('embeddings')
    elif args.dataset:
        if args.dataset.lower() == 'all':
            downloader.download_dataset_group('all_datasets')
        else:
            downloader.download_single_dataset(args.dataset)
    else:
        print("No action specified. Use --help for usage information.")
        parser.print_help()


if __name__ == '__main__':
    main()
