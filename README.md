# RNAG- Retrieval Non Augumented Generation


# Retrieval Non Augmented Generation (RNAG)

## Overview

Retrieval Non Augmented Generation (RNAG) is an innovative approach in natural language processing that refrains from conventional data augmentation methods used in many generative models. Instead, RNAG focuses on directly answering queries based solely on internal data, thus eliminating the need for incorporating external information sources like retrieval-augmented generation techniques.

### Key Features:
- **Non-Augmentation**: Unlike RAG variants that stand on data augmentation, RNAG leverages internal logical constructs to make inferences and generate responses.
- **Direct Query Handling**: The approach utilizes input queries as they are, fostering quicker response times by sidestepping complex retrieval processes.
- **Session Management**: Organizes dialogue through defined sessions, maintaining state (queries, responses) across interactions.
- **Embedding and Similarity**: Employs embedding techniques to assess similarity between queries and stored contexts, ensuring relevant responses.

## Implementation Details

The core implementation of RNAG as demonstrated in the provided code showcases handling incoming queries through various functions. Here are a few components:

### JsonDB Class
Handles the storage and retrieval of queries and their corresponding responses using embeddings for similarity comparison.

### Session Class
Manages individual user sessions, storing conversations and handling offline queries.

### MainProcessWorkflow Class
Coordinates the overall workflow, managing sessions, handling queries, and processing offline queries.

#### Key Methods:
- **start_new_session**: Initializes a new session.
- **handle_query**: Processes incoming queries, retrieves or generates responses based on session data.
- **offline_cosine_similarity**: Computes similarity between queries and stored data when offline.
- **process_offline_queries**: Processes stored offline queries when back online.
- **save_session_to_json**: Saves session data to a JSON file.
- **load_session_from_json**: Loads session data from a JSON file.
- **load_all_sessions**: Loads all session data from the session folder.
- **clear_memory**: Clears unused variables to free up memory.
- **reinitialize_models**: Reinitializes models to ensure they are up-to-date.

## Setup and Installation

### Prerequisites
- Python 3.7+
- Hugging Face Token

### Installation Steps

1. **Clone the repository:**
    ```sh
    git clone <repository_url>
    cd <repository_directory>
    ```

2. **Install the required packages:**
    ```sh
    pip install -r requirements.txt
    ```

3. **Set the Hugging Face Token:**
    ```sh
    export HUGGINGFACE_TOKEN=<your_hugging_face_token>
    ```

4. **Run the Streamlit application:**
    ```sh
    streamlit run app.py
    ```

### Running in Google Colab

1. **Install LocalTunnel:**
    ```sh
    !npm install localtunnel
    ```

2. **Get IP Address for LocalTunnel:**
    ```python
    import urllib
    print("Password/Endpoint IP for LocalTunnel is:", urllib.request.urlopen('https://ipv4.icanhazip.com').read().decode('utf8').strip("\n"))
    ```

3. **Run Streamlit and LocalTunnel:**
    ```sh
    !streamlit run app.py &>/content/logs.txt & npx localtunnel --port 8501
    ```

4. **Access the application:**
    - A link will be displayed. Click it and paste the IP address of LocalTunnel in it.
    - Let the code run.

## Example Output

To illustrate the expected output from the model:





**NEW SESSION**

![NEW SESSION](https://lh3.googleusercontent.com/pw/AP1GczORSVgIszggWiEfsWEfH7es4wkt02cVeqnLsBhZjCZirihgpWvHo5vdIzRl5XQpx0Nvcfu2UQmJOlXnllO1iE8M0jEFMueOmbM5VhJvoSDJLiVor4HbTfqxjIaGISSEDpvnkhHvHODAWfV7JzFVQ06L=w1915-h898-s-no-gm)

**CODE LIBRARIES**

![CODE LIBRARIES](https://lh3.googleusercontent.com/pw/AP1GczPe9kXYacU2r8Bgq-v1jnZ7bFn0JbvLvJkkgxz1k9zUyan9lunX1RLtKZIi9rukqU_M9JhE66J6fRvQS6p6Cq-vc_Xhd257en-_dDHHn_8e_ItQ8lPdV6BZaNXIVg-kMXqDxC7ftDnqp4xD-8iWkisQ=w1513-h825-s-no-gm)

**TESTING**

![TESTING](https://lh3.googleusercontent.com/pw/AP1GczO0wzjK-fAff-239hL_eDRky3nEw8Fe0iNLBpQWj6o8iiAWQssTQZx5wjTc4pkof78omwma0SVVh40U9KuATblFHVbyEpc33ajYUa53GG2rL0u25eoCry2v-v4rlnHgcCYUs2_Xo53KLqgXDW-xaIfc=w1514-h835-s-no-gm)

**METRICS**

![METRICS](https://lh3.googleusercontent.com/pw/AP1GczOjo-AwaDpXx0QXGW3O9CWcPA3DYGscknEHiEsxxr1nUprClbiqGlZVyYuABy0h3Q3KehMJFnzYfxQLIunYjXtzgwvR80CoOPlo-ufZeQ_K_bLOsuzFZQ4DoTzZBbCvVFa3knOijjAIWPxDIcY-6IMK=w1475-h805-s-no-gm)

**CHAT IN SESSIONS**

![CHAT IN SESSIONS:](https://lh3.googleusercontent.com/pw/AP1GczNCUaqqTFjKllM-WcN2gBzu5hlH-OTpWyWj29aebjxOz51wlPo5B2WUHNp7ooIEvReRT92FDodwaFUv4EgkvblSBoXBEkFcAJ8-LifB-QDKk4GLhi6r7zfEcOPngCJEwVbxk6cCQK4ddCgWQN2ZeXnK=w1913-h887-s-no-gm)
