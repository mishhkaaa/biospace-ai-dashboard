# This script loads entities and relations from CSV files into a Neo4j Aura database.
#
# Before running:
# 1. Make sure you have the required libraries:
#    pip install neo4j pandas python-dotenv
# 2. Create a .env file with your Neo4j Aura credentials (see .env.example)
# 3. Make sure entities.csv and relations.csv are in the results/ folder.

import pandas as pd
from neo4j import GraphDatabase, exceptions
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class KnowledgeGraphLoader:
    def __init__(self, uri, user, password):
        """Initializes the loader and connects to the Neo4j database."""
        try:
            self.driver = GraphDatabase.driver(uri, auth=(user, password))
            self.driver.verify_connectivity()
            print("✅ Successfully connected to Neo4j Aura.")
        except exceptions.AuthError as e:
            print(f"❌ Authentication failed: {e}. Please check your credentials.")
            self.driver = None
        except Exception as e:
            print(f"❌ Failed to connect to Neo4j: {e}")
            self.driver = None

    def close(self):
        """Closes the database connection."""
        if self.driver:
            self.driver.close()
            print("✅ Database connection closed.")

    def drop_constraints(self, tx):
        """Drops all existing constraints."""
        constraints = tx.run("SHOW CONSTRAINTS").data()
        for constraint in constraints:
            tx.run(f"DROP CONSTRAINT `{constraint['name']}`")
    
    def clear_database(self, tx):
        """
        DANGER: This will delete everything in the database.
        It's useful for clean re-runs during development.
        """
        print("🗑️  Clearing the database...")
        # Delete all nodes and relationships.
        tx.run("MATCH (n) DETACH DELETE n")
        print("✅ Database cleared.")

    def create_constraints(self, tx, entities_file):
        """Creates uniqueness constraints to improve performance and ensure data integrity."""
        print("🔧 Creating constraints...")
        df_entities = pd.read_csv(entities_file)
        unique_labels = df_entities['Entity_Type'].unique()
        for label in unique_labels:
            # Cypher is case-sensitive with labels, so we ensure clean label names.
            # Property name is 'name' for all nodes.
            query = f"CREATE CONSTRAINT IF NOT EXISTS FOR (n:`{label}`) REQUIRE n.name IS UNIQUE"
            tx.run(query)
        print(f"✅ Constraints created for {len(unique_labels)} entity types: {', '.join(unique_labels)}")

    def load_nodes(self, tx, entities_file):
        """Loads nodes from entities.csv with CSV column properties."""
        print("📥 Loading nodes from entities...")
        df_entities = pd.read_csv(entities_file)
        
        # Show file statistics
        print(f"   Total entity records in CSV: {len(df_entities)}")
        
        # Each row becomes a node with all CSV columns as properties
        # Group by unique entity text to avoid duplicates
        df_unique = df_entities.drop_duplicates(subset=['Entity_Text', 'Entity_Type'])
        print(f"   Unique entities to load: {len(df_unique)}")
        
        # Count by type
        entity_counts = df_unique['Entity_Type'].value_counts()
        print(f"   Entity distribution:")
        for entity_type, count in entity_counts.items():
            print(f"      {entity_type}: {count}")

        # Load each entity with CSV properties
        batch_size = 50
        for i, (_, row) in enumerate(df_unique.iterrows()):
            # Create node with exact CSV columns
            query = f"""
            MERGE (n:`{row['Entity_Type']}` {{name: $entity_text}})
            SET n.PMCID = $pmcid,
                n.Entity_Text = $entity_text,
                n.Entity_Type = $entity_type,
                n.Span_Start = $span_start,
                n.Span_End = $span_end
            """
            tx.run(
                query,
                entity_text=row['Entity_Text'],
                pmcid=row['PMCID'],
                entity_type=row['Entity_Type'],
                span_start=int(row['Span_Start']),
                span_end=int(row['Span_End'])
            )
            
            # Progress indicator
            if (i + 1) % batch_size == 0:
                print(f"   Processed {i + 1}/{len(df_unique)} nodes...")
        
        print(f"✅ {len(df_unique)} unique nodes loaded successfully.")

    def load_relationships(self, tx, relations_file):
        """Loads relationships from relations.csv with CSV column properties."""
        print("🔗 Loading relationships...")
        df_relations = pd.read_csv(relations_file)
        
        print(f"   Total relationships to load: {len(df_relations)}")
        
        # Show relationship distribution
        predicate_counts = df_relations['Predicate'].value_counts()
        print(f"   Relationship distribution:")
        for predicate, count in predicate_counts.items():
            print(f"      {predicate}: {count}")

        # Remove duplicate relationships (same subject-predicate-object)
        df_unique = df_relations.drop_duplicates(
            subset=['Subject_Text', 'Subject_Type', 'Predicate', 'Object_Text', 'Object_Type']
        )
        print(f"   Unique relationships to load: {len(df_unique)}")

        # Load relationships with CSV properties
        batch_size = 100
        for i, (_, row) in enumerate(df_unique.iterrows()):
            # Create relationship with exact CSV columns
            query = """
            MATCH (subject:`{subject_type}` {{name: $subject_text}})
            MATCH (object:`{object_type}` {{name: $object_text}})
            MERGE (subject)-[r:`{predicate}`]->(object)
            SET r.PMCID = $pmcid,
                r.Subject_Text = $subject_text,
                r.Subject_Type = $subject_type,
                r.Predicate = $predicate,
                r.Object_Text = $object_text,
                r.Object_Type = $object_type
            """.format(
                subject_type=row['Subject_Type'],
                object_type=row['Object_Type'],
                predicate=row['Predicate']
            )
            tx.run(
                query, 
                subject_text=row['Subject_Text'],
                object_text=row['Object_Text'],
                pmcid=row['PMCID'],
                subject_type=row['Subject_Type'],
                predicate=row['Predicate'],
                object_type=row['Object_Type']
            )
            
            # Progress indicator
            if (i + 1) % batch_size == 0:
                print(f"   Processed {i + 1}/{len(df_unique)} relationships...")
        
        print(f"✅ {len(df_unique)} unique relationships loaded successfully.")

    def get_graph_stats(self, tx):
        """Retrieves and displays statistics about the loaded graph."""
        print("\n📊 Knowledge Graph Statistics:")
        
        # Count nodes by label
        node_query = """
        MATCH (n)
        RETURN labels(n)[0] as label, count(n) as count
        ORDER BY count DESC
        """
        node_results = tx.run(node_query).data()
        print(f"   Total Nodes: {sum(r['count'] for r in node_results)}")
        for result in node_results:
            print(f"      {result['label']}: {result['count']}")
        
        # Count relationships by type
        rel_query = """
        MATCH ()-[r]->()
        RETURN type(r) as type, count(r) as count
        ORDER BY count DESC
        """
        rel_results = tx.run(rel_query).data()
        print(f"   Total Relationships: {sum(r['count'] for r in rel_results)}")
        for result in rel_results:
            print(f"      {result['type']}: {result['count']}")
        
        # Find most connected nodes
        print("\n🔗 Most Connected Entities:")
        top_nodes_query = """
        MATCH (n)
        RETURN labels(n)[0] as Type, n.name as Entity, COUNT { (n)--() } as Connections
        ORDER BY Connections DESC
        LIMIT 5
        """
        top_nodes = tx.run(top_nodes_query).data()
        for node in top_nodes:
            print(f"      {node['Type']}: '{node['Entity']}' - {node['Connections']} connections")

    def run_loader(self, entities_file, relations_file, clear_db=True):
        """Executes the full loading pipeline."""
        if not self.driver:
            print("❌ Cannot run loader, no valid database connection.")
            return

        # Check if files exist
        if not os.path.exists(entities_file):
            print(f"❌ Error: Entities file not found: {entities_file}")
            return
        if not os.path.exists(relations_file):
            print(f"❌ Error: Relations file not found: {relations_file}")
            return

        print(f"\n🚀 Starting knowledge graph loading pipeline...")
        print(f"   Entities file: {entities_file}")
        print(f"   Relations file: {relations_file}")
        print()

        with self.driver.session() as session:
            # Using transactions ensures that each step completes successfully.
            if clear_db:
                # Drop constraints first (separate transaction)
                session.execute_write(self.drop_constraints)
                # Then clear the database
                session.execute_write(self.clear_database)
            session.execute_write(self.create_constraints, entities_file)
            session.execute_write(self.load_nodes, entities_file)
            session.execute_write(self.load_relationships, relations_file)
            session.execute_read(self.get_graph_stats)
        
        print("\n🎉 Knowledge graph loading complete!")
        print("\n💡 Next steps:")
        print("   1. Open your Neo4j Aura browser")
        print("   2. Try this query to visualize the graph:")
        print("      MATCH (n)-[r]->(m) RETURN n, r, m LIMIT 50")
        print("   3. Explore specific entities:")
        print("      MATCH (n:ORGANISM) RETURN n LIMIT 10")


if __name__ == "__main__":
    # Load credentials from .env file
    AURA_URI = os.getenv("NEO4J_URI")
    AURA_USER = os.getenv("NEO4J_USER", "neo4j")  # Default to 'neo4j' if not set
    AURA_PASSWORD = os.getenv("NEO4J_PASSWORD")
    
    # File paths - update these if your CSV files are in a different location
    RESULTS_FOLDER = "results"
    ENTITIES_FILE = os.path.join(RESULTS_FOLDER, "entities.csv")
    RELATIONS_FILE = os.path.join(RESULTS_FOLDER, "relations.csv")
    
    # Set to False if you want to add to existing data instead of clearing
    CLEAR_DATABASE = True

    # Check if credentials are configured
    if not AURA_URI or not AURA_PASSWORD:
        print("=" * 70)
        print("!!! ERROR: Neo4j Aura credentials not found !!!")
        print("=" * 70)
        print()
        print("Please create a .env file with your Neo4j Aura credentials:")
        print()
        print("   NEO4J_URI=neo4j+s://xxxxx.databases.neo4j.io")
        print("   NEO4J_USER=neo4j")
        print("   NEO4J_PASSWORD=your-password-here")
        print()
        print("You can copy .env.example to .env and update the values.")
        print("Find your credentials in the 'Connect' tab of your Neo4j Aura instance.")
        print("=" * 70)
    else:
        loader = KnowledgeGraphLoader(AURA_URI, AURA_USER, AURA_PASSWORD)
        try:
            loader.run_loader(ENTITIES_FILE, RELATIONS_FILE, clear_db=CLEAR_DATABASE)
        finally:
            loader.close()
