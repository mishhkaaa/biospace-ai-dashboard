"""
NASA Bioscience AI Pipeline - Main Orchestrator
Comprehensive pipeline execution with monitoring, error handling, and reporting
"""

import os
import sys
import time
import traceback
from datetime import datetime
from typing import Dict, List, Optional, Any
import argparse

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from config import config
from utils import PipelineLogger, generate_pipeline_report
from summarization import main as run_summarization
from embeddings import main as run_embeddings
from clustering import main as run_clustering
from keywords import main as run_keywords
from insights import main as run_insights

class NASAPipelineOrchestrator:
    """
    Main orchestrator for the NASA Bioscience AI Pipeline
    """
    
    def __init__(self, skip_components: Optional[List[str]] = None, verbose: bool = True):
        self.logger = PipelineLogger("NASA Pipeline Orchestrator", 
                                   os.path.join(config.paths.base_dir, "pipeline.log"))
        self.skip_components = skip_components or []
        self.verbose = verbose
        self.results = {}
        self.start_time = None
        
        # Pipeline components in execution order
        self.components = {
            'summarization': {
                'name': 'Text Summarization',
                'function': run_summarization,
                'description': 'Generate abstractive summaries of scientific papers',
                'dependencies': [],
                'outputs': ['paper_summaries.csv']
            },
            'embeddings': {
                'name': 'Semantic Embeddings',
                'function': run_embeddings,
                'description': 'Create semantic embeddings for similarity analysis',
                'dependencies': ['summarization'],
                'outputs': ['paper_embeddings.jsonl', 'paper_embeddings_metadata.csv']
            },
            'clustering': {
                'name': 'Document Clustering',
                'function': run_clustering,
                'description': 'Group papers into thematic clusters',
                'dependencies': ['embeddings'],
                'outputs': ['paper_clusters.csv', 'cluster_analysis.json']
            },
            'keywords': {
                'name': 'Keyword Extraction & Cluster Summarization',
                'function': run_keywords,
                'description': 'Extract keywords and create cluster-level summaries',
                'dependencies': ['summarization', 'clustering'],
                'outputs': ['cluster_keywords_summaries.csv']
            },
            'insights': {
                'name': 'Insights Generation',
                'function': run_insights,
                'description': 'Generate actionable insights and research opportunities',
                'dependencies': ['summarization', 'clustering', 'keywords'],
                'outputs': ['knowledge_gaps.json', 'publication_trends.csv', 'consensus_disagreement.csv', 'research_opportunities.json']
            }
        }
    
    def validate_environment(self) -> bool:
        """
        Validate that the environment is properly set up
        
        Returns:
            bool: True if environment is valid
        """
        self.logger.info("Validating pipeline environment")
        
        try:
            # Check if input data exists
            input_file = config.get_data_file_path("person_a_metadata.csv", "raw")
            if not os.path.exists(input_file):
                self.logger.error(f"Input data file not found: {input_file}")
                return False
            
            # Check if output directories exist (create if needed)
            config._create_directories()
            
            # Test imports
            import torch
            import transformers
            import sentence_transformers
            import sklearn
            import keybert
            
            self.logger.success("Environment validation passed")
            return True
            
        except ImportError as e:
            self.logger.error(f"Missing required package: {str(e)}")
            return False
        except Exception as e:
            self.logger.error(f"Environment validation failed: {str(e)}")
            return False
    
    def check_dependencies(self, component: str) -> bool:
        """
        Check if dependencies for a component are satisfied
        
        Args:
            component: Component name
            
        Returns:
            bool: True if dependencies are satisfied
        """
        dependencies = self.components[component]['dependencies']
        
        for dep in dependencies:
            if dep in self.skip_components:
                self.logger.warning(f"Dependency {dep} was skipped but is required for {component}")
                return False
            
            if dep not in self.results or not self.results[dep].get('success', False):
                self.logger.error(f"Dependency {dep} failed or was not executed for {component}")
                return False
        
        return True
    
    def run_component(self, component_name: str) -> Dict[str, Any]:
        """
        Run a single pipeline component with error handling and monitoring
        
        Args:
            component_name: Name of component to run
            
        Returns:
            Dictionary with execution results
        """
        component = self.components[component_name]
        
        self.logger.info(f"=" * 60)
        self.logger.info(f"🚀 Starting {component['name']}")
        self.logger.info(f"Description: {component['description']}")
        self.logger.info(f"=" * 60)
        
        start_time = time.time()
        
        try:
            # Check dependencies
            if not self.check_dependencies(component_name):
                raise RuntimeError(f"Dependencies not satisfied for {component_name}")
            
            # Run component
            result = component['function']()
            
            duration = time.time() - start_time
            
            # Verify outputs exist
            missing_outputs = []
            for output_file in component['outputs']:
                # Determine output type from file extension and component
                if component_name == 'summarization':
                    output_path = config.get_output_file_path(output_file, 'summaries')
                elif component_name == 'embeddings':
                    output_path = config.get_output_file_path(output_file, 'embeddings')
                elif component_name in ['clustering', 'keywords']:
                    output_path = config.get_output_file_path(output_file, 'clusters')
                elif component_name == 'insights':
                    output_path = config.get_output_file_path(output_file, 'insights')
                else:
                    continue
                
                if not os.path.exists(output_path):
                    missing_outputs.append(output_file)
            
            if missing_outputs:
                self.logger.warning(f"Expected outputs not found: {missing_outputs}")
            
            self.logger.success(f"✅ {component['name']} completed successfully in {duration:.2f} seconds")
            
            return {
                'success': True,
                'duration': f"{duration:.2f}s",
                'result': result,
                'output_files': component['outputs'],
                'missing_outputs': missing_outputs
            }
            
        except Exception as e:
            duration = time.time() - start_time
            error_msg = str(e)
            
            self.logger.error(f"❌ {component['name']} failed after {duration:.2f} seconds")
            self.logger.error(f"Error: {error_msg}")
            
            if self.verbose:
                self.logger.error(f"Traceback: {traceback.format_exc()}")
            
            return {
                'success': False,
                'duration': f"{duration:.2f}s",
                'error': error_msg,
                'traceback': traceback.format_exc()
            }
    
    def run_full_pipeline(self) -> Dict[str, Any]:
        """
        Run the complete pipeline with monitoring and reporting
        
        Returns:
            Dictionary with complete execution results
        """
        self.start_time = time.time()
        
        self.logger.info("🔬 NASA BIOSCIENCE AI PIPELINE STARTING")
        self.logger.info(f"Pipeline Configuration:")
        self.logger.info(f"  - Summarization Model: {config.models.summarization_model}")
        self.logger.info(f"  - Embedding Model: {config.models.embedding_model}")
        self.logger.info(f"  - Clustering Algorithm: {config.clustering.clustering_algorithm}")
        self.logger.info(f"  - Default Clusters: {config.clustering.default_n_clusters}")
        self.logger.info(f"  - Skip Components: {self.skip_components}")
        
        # Validate environment
        if not self.validate_environment():
            self.logger.error("Environment validation failed - aborting pipeline")
            return {'success': False, 'error': 'Environment validation failed'}
        
        # Execute components in order
        for component_name in self.components.keys():
            if component_name in self.skip_components:
                self.logger.info(f"⏭️  Skipping {component_name}")
                self.results[component_name] = {'success': True, 'skipped': True}
                continue
            
            self.results[component_name] = self.run_component(component_name)
            
            # Stop if component failed and is critical
            if not self.results[component_name]['success']:
                # For now, treat all components as critical
                self.logger.error(f"Critical component {component_name} failed - stopping pipeline")
                break
        
        # Calculate total duration
        total_duration = time.time() - self.start_time
        
        # Generate comprehensive report
        self.logger.info("📊 Generating pipeline report")
        report = generate_pipeline_report(self.results)
        
        # Save report
        report_path = os.path.join(config.paths.base_dir, f"pipeline_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")
        with open(report_path, 'w') as f:
            f.write(report)
        
        # Final summary
        successful_components = sum(1 for r in self.results.values() if r.get('success', False))
        total_components = len([c for c in self.components.keys() if c not in self.skip_components])
        
        if successful_components == total_components:
            self.logger.success(f"🎉 PIPELINE COMPLETED SUCCESSFULLY!")
            self.logger.success(f"Total duration: {total_duration:.2f} seconds")
            self.logger.success(f"Report saved to: {report_path}")
        else:
            self.logger.error(f"❌ PIPELINE COMPLETED WITH ERRORS")
            self.logger.error(f"Successful: {successful_components}/{total_components} components")
            self.logger.error(f"Total duration: {total_duration:.2f} seconds")
        
        return {
            'success': successful_components == total_components,
            'total_duration': total_duration,
            'successful_components': successful_components,
            'total_components': total_components,
            'results': self.results,
            'report_path': report_path
        }
    
    def run_single_component(self, component_name: str) -> Dict[str, Any]:
        """
        Run a single component (useful for debugging or partial runs)
        
        Args:
            component_name: Name of component to run
            
        Returns:
            Dictionary with execution results
        """
        if component_name not in self.components:
            raise ValueError(f"Unknown component: {component_name}")
        
        self.logger.info(f"Running single component: {component_name}")
        
        # Validate environment
        if not self.validate_environment():
            return {'success': False, 'error': 'Environment validation failed'}
        
        result = self.run_component(component_name)
        
        self.logger.info(f"Single component execution completed")
        return result

def main():
    """Main entry point for the pipeline"""
    parser = argparse.ArgumentParser(description='NASA Bioscience AI Pipeline')
    parser.add_argument('--component', type=str, help='Run single component (summarization, embeddings, clustering, keywords, insights)')
    parser.add_argument('--skip', nargs='+', help='Skip specific components')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose logging')
    parser.add_argument('--config-model', type=str, help='Override summarization model')
    
    args = parser.parse_args()
    
    # Override configuration if specified
    if args.config_model:
        config.models.summarization_model = args.config_model
    
    # Initialize orchestrator
    orchestrator = NASAPipelineOrchestrator(
        skip_components=args.skip or [],
        verbose=args.verbose
    )
    
    try:
        if args.component:
            # Run single component
            result = orchestrator.run_single_component(args.component)
            exit_code = 0 if result['success'] else 1
        else:
            # Run full pipeline
            result = orchestrator.run_full_pipeline()
            exit_code = 0 if result['success'] else 1
        
        sys.exit(exit_code)
        
    except KeyboardInterrupt:
        orchestrator.logger.error("Pipeline interrupted by user")
        sys.exit(130)
    except Exception as e:
        orchestrator.logger.error(f"Pipeline failed with unexpected error: {str(e)}")
        if args.verbose:
            orchestrator.logger.error(f"Traceback: {traceback.format_exc()}")
        sys.exit(1)

if __name__ == "__main__":
    main()