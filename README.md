
### Overview of the Application 

This is a Streamlit-based web application that provides an interactive chat interface for exploring biodiversity data, with a particular focus on endangered species. It uses Google's Vertex AI Gemini model for natural language processing and integrates with various biodiversity data sources.


### Key Features

1. **Interactive Chat Interface**
   - Users can ask questions about endangered species and biodiversity in natural language
   - The system responds with data-driven insights and visualizations


2. **Data Sources**
   - BigQuery database containing endangered species information
   - GBIF (Global Biodiversity Information Facility) occurrence data
   - IUCN Red List conservation status data


3. **Visualization Capabilities**
   - Interactive maps showing species distributions
   - Heatmaps of species occurrences
   - Country-specific geographical data overlays

### Technical Architecture
   - `BiodiversityApp`: Main application class managing the chat interface and model interactions
   - `FunctionHandler`: Manages data retrieval and processing functions
   - `ChartHandler`: Handles data visualization using PyDeck


2. **AI Integration**
   - Uses Vertex AI's Gemini model for natural language understanding
   - Function calling capability to translate user queries into specific data operations


3. **Data Flow**
```
User Query → Gemini Model → Function Calls → BigQuery/APIs → Data Processing → Visualization/Response
```

4. **Key Technologies**
   - Frontend: Streamlit
   - AI: Google Vertex AI (Gemini)
   - Database: Google BigQuery
   - Visualization: PyDeck
   - Cloud Platform: Google Cloud Run (for deployment)


### Deployment
- Can be run locally using Docker
- Deployable to Google Cloud Run for production use
- Requires various Google Cloud APIs and permissions
- Uses secret management for API keys and credentials


### Use Cases

1. Query endangered species information
2. Explore species distributions geographically
3. Analyze conservation status statistics
4. Investigate biodiversity patterns by taxonomic groups
5. View species occurrence data by country or region


This application serves as a powerful tool for researchers, conservationists, and anyone interested in exploring biodiversity data through a user-friendly, conversational interface.


#### Here are several potential enhancements for the Biodiversity Chat application:

1. **Expanded Species Coverage**
   - Currently only includes mammals
   - Add other taxonomic groups (birds, reptiles, amphibians, plants)
   - Include non-endangered species for comparison


2. **Temporal Data Analysis**
   - Add historical occurrence data
   - Track population changes over time
   - Monitor conservation status changes


3. **Environmental Data Integration**
   - Climate data correlation
   - Habitat type mapping
   - Human impact indicators
   - Protected areas overlay


### Technical Doucmentation

# Configuration

## Secrets Management

This application uses Streamlit's secrets management system to handle sensitive configuration values. The secrets are stored in `.streamlit/secrets.toml`.

### Required API Keys and Secrets

This application requires several API keys and configuration values to be set in `.streamlit/secrets.toml`:

- `GOOGLE_CLOUD_PROJECT`: Your Google Cloud Project ID used for Vertex AI initialization


1. **Google Search API Configuration**
   - `GOOGLE_API_KEY`: Your Google API key for accessing Google Custom Search
   - `GOOGLE_CSE_ID`: Your Google Custom Search Engine ID
   
### Local Development

1. Create a `.streamlit/secrets.toml` file in your project root:
   ```toml
   GOOGLE_CLOUD_PROJECT = "your-project-id"
   GOOGLE_API_KEY = "your-google-api-key"
   GOOGLE_CSE_ID = "your-custom-search-engine-id"
   ```
2. Add `.streamlit/secrets.toml` to your `.gitignore` file
3. Never commit secrets to version control

#### Permissions needed

# Enable the following API for you google project:
1. VertexAI API
2. BigQuery API
(Additionally, if you deploy on google cloud)
3. Secret Manager API 
4. Cloud Build API 
5. Cloud Run Admin API 

# Running the application locally 
streamlit run app.py 

you have to be logged in to google cloud and have to select the relevant cloud project 

# Running the application locally in a Docker container

1. add your credentials.json key file from your service principal to .streamlit/credentials.json  
2. build the docker image using Dockerfile_local, it copies the credentials into the docker image
      docker build -f Dockerfile_local -t biochat .
3. Run the application: 
      docker run  -p 8888:8080 biochat
4. Access the application under http://localhost:8888 


### Data preparations
The application access endangered species data and GBIF occurance data from Google BigQuery. 
Since billing for BigQuery is based on the bytes the query scans and the GBIF occurance data is about 1.5 TB big, we copy over part of the occurance data, data only for endangered species and only for mammals. 

1. Create a schema 'biodiversity' under your Google project in BigQuery. 
2. run the script scripts/convert.py and scripts/convert_dist.py to convert the two files taxon.txt and distribution.txt to the corresponding csv files (*)
3. import the two csv files to the biodiversity schema distribution.csv -> table conservation_status, taxon.csv -> table endangered_species
4. Run the following query to transfer occurances from GBIF occurance data to the biodiversity schema: 

CREATE OR REPLACE TABLE `[your_project_id].biodiversity.occurances_endangered_species` AS
SELECT  species, decimallongitude, decimallatitude, countrycode, individualcount, eventdate  
FROM `bigquery-public-data.gbif.occurrences` 
WHERE decimallatitude is not null 
  AND decimallongitude is not null 
  AND species IN (
    SELECT CONCAT(genus_name, ' ', species_name) as full_name 
    FROM `[your_project_id].biodiversity.endangered_species` 
    WHERE species_name is not null 
  )

5. Create a column conservation_status in table endangered_species 
6. Runthe following update to update the newly created column: 
UPDATE `[your_project_id].biodiversity.endangered_species` es
SET conservation_status = cs.Scope 
FROM `[your_project_id].biodiversity.conservation_status` cs
WHERE es.conservation_status = CAST(cs.id AS STRING);

 
(*) IUCN (2022). The IUCN Red List of Threatened Species. Version 2022-2. https://www.iucnredlist.org. Downloaded on 2023-05-09. https://doi.org/10.15468/0qnb58 accessed via GBIF.org on 2023-11-17. accessed via GBIF.org on 2024-12-28.

### Deploying and running the application in Production on Google Cloud Run

# Build docker image 
export PROJECT_ID="[your-project-id]"
export PROJECT_ID="tribal-logic-351707"
docker build --platform linux/amd64 -t gcr.io/$PROJECT_ID/biochat-app:latest .

# Push docker image to Google repository
docker push  gcr.io/$PROJECT_ID/biochat-app:latest  


# Create the secrets
gcloud secrets create GOOGLE_CLOUD_PROJECT --data-file=- <<< "[your-project-id]"
gcloud secrets create GOOGLE_API_KEY --data-file=- <<< "[your-google-api-key]"
gcloud secrets create GOOGLE_CSE_ID --data-file=- <<< "[your-cse-id]"

# Grant Secret Manager access to you service principal 
export SERVICE_ACCOUNT="[your-service-principal]"
export SERVICE_ACCOUNT="153810785966-compute@developer.gserviceaccount.com"

gcloud secrets add-iam-policy-binding GOOGLE_API_KEY \
    --member="serviceAccount:$SERVICE_ACCOUNT" \
    --role="roles/secretmanager.secretAccessor"

gcloud secrets add-iam-policy-binding GOOGLE_CSE_ID \
    --member="serviceAccount:$SERVICE_ACCOUNT" \
    --role="roles/secretmanager.secretAccessor"


export SERVICE_ACCOUNT="153810785966-compute@developer.gserviceaccount.com"
gcloud secrets add-iam-policy-binding GOOGLE_CLOUD_PROJECT \
    --member="serviceAccount:$SERVICE_ACCOUNT" \
    --role="roles/secretmanager.secretAccessor"


# Grant Vertex AI User role to the service principal
gcloud projects add-iam-policy-binding $PROJECT_ID \
    --member="serviceAccount:$SERVICE_ACCOUNT" \
    --role="roles/aiplatform.user"   

# Grant BigQuery Job User role (for creating jobs)
gcloud projects add-iam-policy-binding $PROJECT_ID \
    --member="serviceAccount:$SERVICE_ACCOUNT" \
    --role="roles/bigquery.jobUser"

# Grant BigQuery Data Viewer role (for reading data)
gcloud projects add-iam-policy-binding $PROJECT_ID \
    --member="serviceAccount:$SERVICE_ACCOUNT" \
    --role="roles/bigquery.dataViewer"


# Deploy application and reference secrets from Google Cloud Run 
gcloud run deploy biochat-app \
  --image gcr.io/$PROJECT_ID/biochat-app:latest \
  --platform managed \
  --region us-central1 \
  --service-account="$SERVICE_ACCOUNT" \
  --set-secrets="GOOGLE_API_KEY=GOOGLE_API_KEY:latest,GOOGLE_CSE_ID=GOOGLE_CSE_ID:latest,GOOGLE_CLOUD_PROJECT=GOOGLE_CLOUD_PROJECT:latest"

# Access the application 
https://biochat-app-153810785966.us-central1.run.app
