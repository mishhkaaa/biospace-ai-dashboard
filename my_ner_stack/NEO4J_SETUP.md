# Neo4j Knowledge Graph Setup Guide

This guide will help you load your NER results into Neo4j Aura to create an interactive knowledge graph.

## Prerequisites

1. **Neo4j Aura Account** (free tier available)
2. **Python packages**: `neo4j` and `pandas`

## Setup Instructions

### Step 1: Install Required Packages

```powershell
pip install neo4j pandas python-dotenv
```

### Step 2: Create Neo4j Aura Database

1. Go to [Neo4j Aura](https://neo4j.com/cloud/aura/)
2. Sign up or log in
3. Click "Create Database" (select Free tier if available)
4. **IMPORTANT**: Save the generated password - you won't see it again!
5. Wait for the database to start (status should be "Running")

### Step 3: Get Your Connection Credentials

1. In Neo4j Aura, click on your database
2. Click the "Connect" button
3. Copy the following information:
   - **URI**: Something like `neo4j+s://xxxxx.databases.neo4j.io`
   - **Username**: Usually `neo4j`
   - **Password**: The password you saved when creating the database

### Step 4: Configure Your .env File

Add your Neo4j credentials to the `.env` file (the same one with your NCBI API key):

```bash
# Add these lines to your existing .env file
NEO4J_URI=neo4j+s://xxxxx.databases.neo4j.io
NEO4J_USER=neo4j
NEO4J_PASSWORD=your-actual-password
```

**Note:** Your `.env` file should now contain both NCBI and Neo4j credentials.

### Step 5: Run the Script

```powershell
python load_graph.py
```

The script will:
- ✅ Connect to Neo4j Aura
- 🗑️ Clear existing data (if CLEAR_DATABASE=True)
- 🔧 Create uniqueness constraints
- 📥 Load 344 unique entities as nodes
- 🔗 Load 2,917 relationships between nodes
- 📊 Display graph statistics

## What Gets Loaded

### Node Types (Labels)
- **ORGANISM** (e.g., "Mice", "human", "cell")
- **ENVIRONMENT** (e.g., "Space", "microgravity", "ISS")
- **EXPERIMENT** (e.g., "research", "study", "test")
- **OUTCOME** (e.g., "effect", "result", "response")

### Relationship Types
- **STUDIED_IN**: Links organisms/experiments to environments
- **HAS_OUTCOME**: Links experiments/environments to outcomes
- **CAUSED_BY**: Links outcomes to experiments/environments

## Exploring Your Knowledge Graph

### Using Neo4j Browser

1. Open your Neo4j Aura instance
2. Click "Open" or "Query" to access the Neo4j Browser
3. Try these Cypher queries:

#### Visualize Sample of the Graph
```cypher
MATCH (n)-[r]->(m) 
RETURN n, r, m 
LIMIT 50
```

#### Find All Organisms
```cypher
MATCH (n:ORGANISM) 
RETURN n.name 
LIMIT 20
```

#### Find What Was Studied in Space
```cypher
MATCH (org:ORGANISM)-[:STUDIED_IN]->(env:ENVIRONMENT {name: "Space"})
RETURN org.name as Organism, env.name as Environment
```

#### Find Experiments and Their Outcomes
```cypher
MATCH (exp:EXPERIMENT)-[:HAS_OUTCOME]->(out:OUTCOME)
RETURN exp.name as Experiment, out.name as Outcome
LIMIT 25
```

#### Find Most Connected Entities
```cypher
MATCH (n)
RETURN labels(n)[0] as Type, n.name as Entity, COUNT { (n)--() } as Connections
ORDER BY Connections DESC
LIMIT 10
```

#### Find Paths Between Two Entities
```cypher
MATCH path = (start {name: "Mice"})-[*1..3]-(end {name: "microgravity"})
RETURN path
LIMIT 10
```

## Customization Options

### Loading Different Files

Edit these lines in `load_graph.py` (around line 189):

```python
ENTITIES_FILE = os.path.join(RESULTS_FOLDER, "entities_20251005_001738.csv")
RELATIONS_FILE = os.path.join(RESULTS_FOLDER, "relations_20251005_001738.csv")
```

### Append Instead of Replace

To add data without clearing the existing graph:

```python
CLEAR_DATABASE = False  # Change from True to False
```

## Troubleshooting

### Authentication Failed
- Double-check your URI, username, and password
- Make sure you're using the password from when you created the database
- Try resetting the password in Neo4j Aura settings

### Connection Timeout
- Check if your Neo4j Aura instance is running
- Verify your internet connection
- Check if firewall is blocking the connection

### File Not Found
- Ensure `entities.csv` and `relations.csv` are in the `results/` folder
- Check the file names match exactly (including timestamp)
- Verify the file paths in the script

## Expected Output

```
✅ Successfully connected to Neo4j Aura.

🚀 Starting knowledge graph loading pipeline...
   Entities file: results/entities_20251005_001738.csv
   Relations file: results/relations_20251005_001738.csv

🗑️  Clearing the database...
✅ Database cleared.
🔧 Creating constraints...
✅ Constraints created for 4 entity types: ORGANISM, ENVIRONMENT, EXPERIMENT, OUTCOME
📥 Loading nodes from entities...
   Total entity records in CSV: 344
   Unique entities to load: 344
   Entity distribution:
      ENVIRONMENT: 100
      ORGANISM: 84
      OUTCOME: 85
      EXPERIMENT: 75
   Processed 50/344 nodes...
   Processed 100/344 nodes...
   ...
✅ 344 unique nodes loaded successfully.
🔗 Loading relationships...
   Total relationships to load: 2917
   Relationship distribution:
      STUDIED_IN: 1500
      HAS_OUTCOME: 1200
      CAUSED_BY: 217
   Processed 100/2917 relationships...
   ...
✅ 2917 relationships loaded successfully.

📊 Knowledge Graph Statistics:
   Total Nodes: 344
      ENVIRONMENT: 100
      ORGANISM: 84
      OUTCOME: 85
      EXPERIMENT: 75
   Total Relationships: 2917
      STUDIED_IN: 1500
      HAS_OUTCOME: 1200
      CAUSED_BY: 217

🎉 Knowledge graph loading complete!
```

## Next Steps

1. Explore the graph visually in Neo4j Browser
2. Run analytical queries to find patterns
3. Export specific subgraphs for further analysis
4. Connect to visualization tools like Neo4j Bloom
5. Process the remaining 597 PMCIDs and reload the graph

## Resources

- [Neo4j Cypher Manual](https://neo4j.com/docs/cypher-manual/current/)
- [Neo4j Aura Documentation](https://neo4j.com/docs/aura/)
- [Graph Data Science Library](https://neo4j.com/docs/graph-data-science/current/)
