import streamlit as st
import os
from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from dotenv import load_dotenv
from gremlin_python.driver import client, serializer
from groq import Groq
from typing import Dict, List
import hashlib
from PIL import Image
import base64


# Configure Streamlit page
st.set_page_config(page_title="Document Search System", layout="wide")

class GroqAnalyzer:
    def __init__(self, api_key: str, model_name: str):
        """Initialize Groq analyzer with API key"""
        self.client = Groq(api_key=api_key)
        self.model_name = model_name

    def generate_summary(self, content: str) -> str:
        """Generate a concise summary of document content"""
        content_words = len(content.split())
        
        if content_words < 500:
            points = "three"
        elif content_words < 1000:
            points = "five"
        else:
            points = "seven"
            
        prompt = f"""Analyze the following document content and provide a {points}-point summary:
        Document Content: {content[:4000]}  # Limit content length
        
        Provide a clear, bullet-point summary that includes {points} main points.
        Make each point concise but informative."""

        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": "You are a document analysis expert that provides clear, structured summaries in bullet points."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=500
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"Error generating summary: {str(e)}"

def get_hash(text):
    """Generate a unique hash for vertex IDs"""
    return hashlib.md5(str(text).encode()).hexdigest()

def get_image_base64(image_path):
    """Convert an image to base64 encoding for inline HTML display"""
    import base64
    import os
    
    # Get the full path
    current_dir = os.path.dirname(os.path.abspath(__file__))
    full_path = os.path.join(current_dir, image_path)
    
    # Encode image
    if os.path.exists(full_path):
        with open(full_path, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode()
    return ""

def init_gremlin_client():
    """Initialize Gremlin client"""
    try:
        gremlin_client = client.Client(
            f'wss://{os.getenv("GREMLIN_HOST")}:{os.getenv("GREMLIN_PORT")}/',
            'g',
            username=f'/dbs/{os.getenv("GREMLIN_DATABASE")}/colls/{os.getenv("GREMLIN_COLLECTION")}',
            password=os.getenv("GREMLIN_PASSWORD"),
            message_serializer=serializer.GraphSONSerializersV2d0()
        )
        return gremlin_client
    except Exception as e:
        st.error(f"Failed to initialize Gremlin client: {str(e)}")
        st.stop()

def init_azure_search():
    """Initialize Azure Search client with proper error handling"""
    # Load environment variables
    load_dotenv()
    
    # Get Azure Search configurations
    search_endpoint = os.getenv("AZURE_SEARCH_SERVICE_ENDPOINT")
    search_key = os.getenv("AZURE_SEARCH_API_KEY")
    index_name = os.getenv("AZURE_SEARCH_INDEX_NAME")
    
    # Validate credentials
    if not all([search_endpoint, search_key, index_name]):
        st.error("""
        Missing Azure Search credentials. Please ensure you have a .env file with:
        - AZURE_SEARCH_ENDPOINT
        - AZURE_SEARCH_KEY
        - AZURE_SEARCH_INDEX_NAME
        """)
        st.stop()
    
    try:
        # Initialize the search client
        credential = AzureKeyCredential(search_key)
        client = SearchClient(endpoint=search_endpoint,
                            index_name=index_name,
                            credential=credential)
        return client
    except Exception as e:
        st.error(f"Failed to initialize Azure Search client: {str(e)}")
        st.stop()

def init_groq():
    """Initialize Groq analyzer"""
    try:
        analyzer = GroqAnalyzer(
            api_key=os.getenv("GROQ_API_KEY"),
            model_name="llama-3.3-70b-versatile"
        )
        return analyzer
    except Exception as e:
        st.error(f"Failed to initialize Groq client: {str(e)}")
        print(f"Detailed error: {e}")  # For debugging
        st.stop()

def init_session_state():
    """Initialize session state variables"""
    if 'search_results' not in st.session_state:
        st.session_state.search_results = None
    if 'selected_doc_id' not in st.session_state:
        st.session_state.selected_doc_id = None
    if 'selected_people' not in st.session_state:
        st.session_state.selected_people = set()
    if 'selected_organizations' not in st.session_state:
        st.session_state.selected_organizations = set()
    if 'selected_locations' not in st.session_state:
        st.session_state.selected_locations = set()
    if 'gremlin_client' not in st.session_state:
        st.session_state.gremlin_client = init_gremlin_client()
    if 'search_client' not in st.session_state:
        st.session_state.search_client = init_azure_search()
    if 'groq_analyzer' not in st.session_state:
        st.session_state.groq_analyzer = init_groq()
    if 'show_similar_docs' not in st.session_state:
        st.session_state.show_similar_docs = False
    if 'similar_docs' not in st.session_state:
        st.session_state.similar_docs = []
    if 'similar_doc_history' not in st.session_state:
        st.session_state.similar_doc_history = []
    if 'current_doc_content' not in st.session_state:
        st.session_state.current_doc_content = None
    if 'document_summaries' not in st.session_state:
        st.session_state.document_summaries = {}
    if 'viewing_document' not in st.session_state:
        st.session_state.viewing_document = False

def search_documents(client, search_text):
    """
    Search documents using Azure Search
    """
    try:
        results = client.search(
            search_text,
            select=[
                # Basic fields
                "DocumentName", "Library", "merged_content", 
                "people", "organizations", "locations",
                
                # General library fields
                "Doc_Type_General", "Date_General", "Remarks_General",
                
                # HR library fields
                "Employee_No_HR", "Department_HR", "Document_Type_HR", 
                "Name_HR", "Date_HR", "Country_HR",
                
                # Florix library fields
                "Document_Type_Florix", "Remarks_Florix",
                
                # DFTROPIO library fields
                "SERIAL_NO_DFTROPIO", "Name_DFTROPIO", "DOB_DFTROPIO",
                "BOOK_CATEGORY_DFTROPIO", "DESCRIPTION_DFTROPIO",
                "VOLUME_NUMBER_DFTROPIO", "SERIAL_RANGE_DFTROPIO", "ACT_NUMBER_DFTROPIO",
                
                # Finance library fields
                "Document_ID_Finance", "Document_Type_Finance", 
                "Date_Finance", "Info_Finance",
                
                # Ayala Annual Report library fields
                "Name_Ayala_Annual_Report", "Year_Ayala_Annual_Report",
                "DocumentType_Ayala_Annual_Report", "Remarks_Ayala_Annual_Report",
                
                # Ayala Legal Docs library fields
                "Name_Ayala_Legal_Docs", "DocumentType_Ayala_Legal_Docs",
                "Remarks_Ayala_Legal_Docs"
            ],
            include_total_count=True
        )
        return list(results)  # Convert to list to make it reusable
    except Exception as e:
        st.error(f"Search failed: {str(e)}")
        return []

def get_related_documents(gremlin_client, selected_people, selected_organizations, selected_locations):
    """Get documents related to selected entities using Gremlin query"""
    try:
        # Convert sets to lists for Gremlin query
        people_list = list(selected_people)
        orgs_list = list(selected_organizations)
        locations_list = list(selected_locations)
        
        # Only proceed if there are selected entities
        if not (people_list or orgs_list or locations_list):
            return []
            
        # Build the OR conditions dynamically based on which lists have items
        or_conditions = []
        if people_list:
            # Using 'peopl' to match your database label
            or_conditions.append(f"has('name', within({str(people_list)})).hasLabel('peopl')")
        if orgs_list:
            or_conditions.append(f"has('name', within({str(orgs_list)})).hasLabel('organization')")
        if locations_list:
            or_conditions.append(f"has('name', within({str(locations_list)})).hasLabel('location')")
            
        # Join the conditions with .or()
        or_clause = ".or(" + ",".join(or_conditions) + ")"
        
        # Construct the complete query
        query = f"""
        g.V()
        .hasLabel('document')
        .where(
            out('mentions')
            {or_clause}
        )
        .project('document', 'library', 'matched_entities')
        .by('name')
        .by(out('belongs_to').values('name'))
        .by(
            out('mentions')
            .group()
            .by('type')
            .by(values('name').fold())
        )
        """
        
        print(f"Executing query: {query}")  # For debugging
        
        # Execute query
        result = gremlin_client.submit(query).all().result()
        
        return result
    except Exception as e:
        st.error(f"Error querying related documents: {str(e)}")
        print(f"Full error: {e}")  # For debugging
        return []

def group_by_library(search_results):
    """
    Group search results by library
    """
    library_groups = {}
    for result in search_results:
        library = result.get('Library', 'Unknown Library')
        if library not in library_groups:
            library_groups[library] = []
        library_groups[library].append(result)
    return library_groups

def display_header():
    """Display the Enadoc logo and AI Document Search title at the top with minimal spacing"""
    # Remove default padding at the top of the page
    st.markdown("""
    <style>
        .block-container {
            padding-top: 0rem;
            padding-bottom: 0rem;
        }
    </style>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        # Extremely tight spacing between logo and title
        st.markdown("""
        <div style="display: flex; flex-direction: column; align-items: center; justify-content: center; margin-bottom: 10px; margin-top: 0px;">
            <img src="data:image/png;base64,{encoded_image}" alt="Enadoc Logo" style="width: 150px; margin-bottom: -25px;">
            <h1 style="margin-top: -5px; text-align: center; line-height: 0.8;">AI Document Search</h1>
        </div>
        """.format(encoded_image=get_image_base64("enadoc_letter_logo.png")), unsafe_allow_html=True)

def apply_table_styles():
    """Apply CSS styles for tables and document view"""
    st.markdown("""
    <style>
    /* Logo styling */
    .logo-container {
        position: absolute;
        top: 10px;
        left: 10px;
        z-index: 1000;
    }
    
    /* Table styles with thick black borders */
    .metadata-table {
        width: 100%;
        border-collapse: collapse;
        border: 2px solid black; /* Add thick black border around entire table */
    }
    .metadata-table th {
        background-color: #f0f2f6;
        font-weight: bold;
        text-align: left;
        padding: 3px 6px;
        border-bottom: 2px solid black; /* Thicker black border for headers */
        border-right: 2px solid black; /* Add vertical borders */
    }
    .metadata-table td {
        padding: 2px 6px;
        border-bottom: 2px solid black; /* Thicker black border for cells */
        border-right: 2px solid black; /* Add vertical borders */
        font-size: 0.9em;
    }
    .metadata-table tr:hover {
        background-color: #f5f5f5;
    }
    
    /* Compact styling for tables */
    .compact-cell {
        font-size: 0.9em;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
        max-width: 150px;
    }
    .view-button {
        text-align: center;
    }
    .compact-text {
        font-size: 1em;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
    }
    
    /* Thick black borders for all tables and dividers */
    hr {
        margin: 1px 0 !important;
        padding: 0 !important;
        border-top: 2px solid black !important; /* Change from #eee to black */
    }
    
    /* Row spacing class */
    .row-spacing {
        margin-top: 0 !important;
        margin-bottom: 0 !important;
        line-height: 1 !important;
        padding: 2px 0 !important;
    }
    
    /* Buttons with thicker borders */
    button.stButton {
        padding: 0 !important;
        height: 24px !important;
        min-height: 24px !important;
        border: 3px solid black !important;
        border-radius: 4px !important;
    }
    
    /* Style for the Find Similar Documents button with thicker border */
    .similar-docs-button-container .stButton button {
        background-color: #8BC34A !important;
        color: white !important;
        font-weight: bold !important;
        border: 3px solid black !important;
    }
    
    .similar-docs-button-container .stButton button:hover {
        background-color: #7CB342 !important;
    }
    
    /* Style for the Clear Entities button */
    button.stButton:contains("Clear Entities") {
        background-color: #FF6B6B !important;
        color: white !important;
        font-weight: bold !important;
        border: 3px solid black !important;
    }
    
    button.stButton:contains("Clear Entities"):hover {
        background-color: #FF5252 !important;
    }
    
    /* Increase font size for general content */
    .stMarkdown p, .stExpander p, .stContainer p {
        font-size: 18px !important;
    }
    
    /* Increase font size for lists */
    .stMarkdown ul, .stMarkdown ol, .stMarkdown li {
        font-size: 18px !important;
    }
    
    /* Search bar styling with thick black border */
    .stTextInput > div > div > input {
        border: 2px solid black !important;
        border-radius: 4px;
    }
    
    /* Add extra thick black borders to expanders */
    .st-expander {
        border: 4px solid black !important;
        margin-bottom: 10px;
    }
    
    /* Document view styles */
    .document-title {
        color: #1E88E5;
        font-size: 28px;
        font-weight: bold;
        padding: 10px 0;
        border-bottom: 2px solid #1E88E5; /* Keeping blue color for this border */
        margin-bottom: 20px;
        display: none; /* Hide the document title */
    }
    
    /* Section title - keeping original blue gradient styling untouched */
    .section-title {
        font-size: 24px;
        font-weight: bold;
        margin-bottom: 15px;
        color: #1E88E5;
        background-color: #f9f9f9;
        padding: 5px 15px;
        border-left: 10px solid #1E88E5;
        border-image: linear-gradient(to bottom, #1E88E5, #64B5F6, #1E88E5) 1 100%;
        display: inline-block;
        border-radius: 5px;
    }
    
    /* Header container with margin to accommodate logo */
    .header-container {
        display: flex;
        justify-content: center;
        align-items: center;
        margin-top: 10px;
        position: relative;
    }

    /* Title and logo container - new styles for proper alignment */
    .title-container {
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        margin-bottom: 20px;
    }
    
    /* Style for the logo with zero bottom margin */
    .title-container img {
        width: 150px;
        margin-bottom: 0;
    }
    
    /* Style for the title with minimal top margin */
    .title-container h1 {
        margin-top: 0;
        text-align: center;
    }
    </style>
    """, unsafe_allow_html=True)

def display_document_content(doc, doc_id, is_similar_view=False):
    """Display document content and entities with selectable checkboxes in organized tiles"""
    # Keep document name in a variable but don't display it
    doc_name = doc.get('DocumentName', 'Untitled Document')
    # Store the document name but hide it with CSS
    st.markdown(f'<div class="document-title" style="display: none;">{doc_name}</div>', unsafe_allow_html=True)
    
    # ----- DOCUMENT SUMMARY SECTION -----
    st.markdown('<div class="document-section">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Document Summary</div>', unsafe_allow_html=True)
    
    # Generate or retrieve document summary
    if doc_name not in st.session_state.document_summaries:
        with st.spinner("Generating document summary..."):
            content = doc.get('merged_content', '')
            summary = st.session_state.groq_analyzer.generate_summary(content)
            st.session_state.document_summaries[doc_name] = summary
    
    # Display the summary
    st.markdown(st.session_state.document_summaries[doc_name])
    
    # Add expander for full content
    with st.expander("View Full Document Content", expanded=False):
        st.write(doc.get('merged_content', 'Content not available'))
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # ----- DOCUMENT METADATA SECTION -----
    st.markdown('<div class="document-section">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Document Metadata</div>', unsafe_allow_html=True)
    
    library = doc.get('Library', 'Unknown')
    
    # Function to create a Streamlit native table from metadata dictionary
    def display_metadata_table(metadata_dict):
        # Create a clean table using Streamlit columns with an empty column to adjust spacing
        for field, value in metadata_dict.items():
            # Add a third column that's empty, which can push the first two closer together
            col1, col2, col3 = st.columns([0.8, 1, 2], gap="small")
            with col1:
                st.markdown(f"**{field}:**")
            with col2:
                st.write(value)
            with col3:
                # Empty column to adjust layout
                pass
        
    if library == "General":
        # Create metadata dictionary
        metadata = {
            "Document Type": doc.get('Doc_Type_General') or "N/A",
            "Date": doc.get('Date_General') or "N/A",
            "Remarks": doc.get('Remarks_General') or "N/A"
        }
        
        # Display metadata as Streamlit table
        display_metadata_table(metadata)
    
    elif library == "HR":
        # Create metadata dictionary
        metadata = {
            "Employee Number": doc.get('Employee_No_HR') or "N/A",
            "Department": doc.get('Department_HR') or "N/A",
            "Document Type": doc.get('Document_Type_HR') or "N/A",
            "Name": doc.get('Name_HR') or "N/A",
            "Date": doc.get('Date_HR') or "N/A",
            "Country": doc.get('Country_HR') or "N/A"
        }
        
        # Display metadata as Streamlit table
        display_metadata_table(metadata)
    
    elif library == "Florix":
        # Create metadata dictionary
        metadata = {
            "Document Type": doc.get('Document_Type_Florix') or "N/A",
            "Remarks": doc.get('Remarks_Florix') or "N/A"
        }
        
        # Display metadata as Streamlit table
        display_metadata_table(metadata)
    
    elif library == "DFTROPIO":
        # Create metadata dictionary
        metadata = {
            "Serial No": doc.get('SERIAL_NO_DFTROPIO') or "N/A",
            "Name": doc.get('Name_DFTROPIO') or "N/A",
            "DOB": doc.get('DOB_DFTROPIO') or "N/A",
            "Book Category": doc.get('BOOK_CATEGORY_DFTROPIO') or "N/A",
            "Description": doc.get('DESCRIPTION_DFTROPIO') or "N/A",
            "Volume Number": doc.get('VOLUME_NUMBER_DFTROPIO') or "N/A",
            "Serial Range": doc.get('SERIAL_RANGE_DFTROPIO') or "N/A",
            "Act Number": doc.get('ACT_NUMBER_DFTROPIO') or "N/A"
        }
        
        # Display metadata as Streamlit table
        display_metadata_table(metadata)
    
    elif library == "Finance":
        # Create metadata dictionary
        metadata = {
            "Document ID": doc.get('Document_ID_Finance') or "N/A",
            "Document Type": doc.get('Document_Type_Finance') or "N/A",
            "Date": doc.get('Date_Finance') or "N/A",
            "Info": doc.get('Info_Finance') or "N/A"
        }
        
        # Display metadata as Streamlit table
        display_metadata_table(metadata)
    
    elif library == "Ayala_Annual_Report":
        # Create metadata dictionary
        metadata = {
            "Name": doc.get('Name_Ayala_Annual_Report') or "N/A",
            "Year": doc.get('Year_Ayala_Annual_Report') or "N/A",
            "Document Type": doc.get('DocumentType_Ayala_Annual_Report') or "N/A",
            "Remarks": doc.get('Remarks_Ayala_Annual_Report') or "N/A"
        }
        
        # Display metadata as Streamlit table
        display_metadata_table(metadata)
    
    elif library == "Ayala_Legal_Docs":
        # Create metadata dictionary
        metadata = {
            "Name": doc.get('Name_Ayala_Legal_Docs') or "N/A",
            "Document Type": doc.get('DocumentType_Ayala_Legal_Docs') or "N/A",
            "Remarks": doc.get('Remarks_Ayala_Legal_Docs') or "N/A"
        }
        
        # Display metadata as Streamlit table
        display_metadata_table(metadata)
    
    else:
        st.write("No specific metadata fields available for this library type.")
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    
    # ----- ENTITIES IN DOCUMENT SECTION -----
    st.markdown('<div class="document-section">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Entities in Document</div>', unsafe_allow_html=True)
    
    # Reset entity selections if viewing a new document in similar view
    if is_similar_view:
        st.session_state.selected_people = set()
        st.session_state.selected_organizations = set()
        st.session_state.selected_locations = set()
    
    # Display extracted entities with checkboxes
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.write("#### People")
        if doc.get('people'):
            people_list = doc['people']
            for person in people_list:
                if st.checkbox(f"👤 {person}", 
                             key=f"person_{doc_id}_{person}",
                             value=person in st.session_state.selected_people):
                    st.session_state.selected_people.add(person)
                else:
                    st.session_state.selected_people.discard(person)
        else:
            st.write("No people mentioned")
    
    with col2:
        st.write("#### Organizations")
        if doc.get('organizations'):
            org_list = doc['organizations']
            for org in org_list:
                if st.checkbox(f"🏢 {org}", 
                             key=f"org_{doc_id}_{org}",
                             value=org in st.session_state.selected_organizations):
                    st.session_state.selected_organizations.add(org)
                else:
                    st.session_state.selected_organizations.discard(org)
        else:
            st.write("No organizations mentioned")
    
    with col3:
        st.write("#### Locations")
        if doc.get('locations'):
            location_list = doc['locations']
            for location in location_list:
                if st.checkbox(f"📍 {location}", 
                             key=f"location_{doc_id}_{location}",
                             value=location in st.session_state.selected_locations):
                    st.session_state.selected_locations.add(location)
                else:
                    st.session_state.selected_locations.discard(location)
        else:
            st.write("No locations mentioned")
    
    # Display currently selected entities (now inside the Entities section)
    st.markdown('<hr style="margin-top: 15px; margin-bottom: 15px;">', unsafe_allow_html=True)
    st.write("#### Currently Selected Entities")
    selected_col1, selected_col2, selected_col3 = st.columns(3)
    
    with selected_col1:
        st.write("**Selected People:**")
        if st.session_state.selected_people:
            for person in st.session_state.selected_people:
                st.write(f"- {person}")
        else:
            st.write("None selected")
            
    with selected_col2:
        st.write("**Selected Organizations:**")
        if st.session_state.selected_organizations:
            for org in st.session_state.selected_organizations:
                st.write(f"- {org}")
        else:
            st.write("None selected")
            
    with selected_col3:
        st.write("**Selected Locations:**")
        if st.session_state.selected_locations:
            for location in st.session_state.selected_locations:
                st.write(f"- {location}")
        else:
            st.write("None selected")
    
    # Create a center-aligned container for both buttons
    st.markdown('<div style="display: flex; justify-content: center; gap: 20px; margin-top: 20px;">', unsafe_allow_html=True)
    
    # Create columns within the centered container
    button_col1, button_col2 = st.columns([1, 1])
    
    # Clear Entities button
    with button_col1:
        if st.button("Clear Entities", key=f"clear_{doc_id}"):
            # Clear all selected entities
            st.session_state.selected_people = set()
            st.session_state.selected_organizations = set()
            st.session_state.selected_locations = set()
            st.experimental_rerun()
    
    # Find Similar Documents button
    with button_col2:
        # Use markdown with custom styling for the button container
        st.markdown('<div class="similar-docs-button-container">', unsafe_allow_html=True)
        if st.button("Find Similar Documents", key=f"similar_{doc_id}"):
            # Only add to history if we have selected entities
            if (st.session_state.selected_people or 
                st.session_state.selected_organizations or 
                st.session_state.selected_locations):
                
                st.session_state.show_similar_docs = True
                st.session_state.viewing_document = False  # Exit document view
                st.session_state.search_results = None  # Clear previous search results
                
                # Add current selections to history
                st.session_state.similar_doc_history.append({
                    'entities': {
                        'people': list(st.session_state.selected_people),
                        'organizations': list(st.session_state.selected_organizations),
                        'locations': list(st.session_state.selected_locations)
                    }
                })
                
                related_docs = get_related_documents(
                    st.session_state.gremlin_client,
                    st.session_state.selected_people,
                    st.session_state.selected_organizations,
                    st.session_state.selected_locations
                )
                
                if related_docs:
                    st.session_state.similar_docs = related_docs
                else:
                    st.session_state.similar_docs = []
                st.experimental_rerun()
            else:
                st.warning("Please select at least one entity to find similar documents.")
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Close the centered container
    st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)  # Close entities section

# -------- Library-Specific Display Functions with Reduced Line Spacing --------

def display_general_library_table(documents):
    """Display General library documents in table format with reduced spacing"""
    # Using Streamlit columns for the table header
    col1, col2, col3, col4 = st.columns([0.5, 2, 1.5, 2.5])
    with col1:
        st.markdown('<p class="row-spacing"><b>View</b></p>', unsafe_allow_html=True)
    with col2:
        st.markdown('<p class="row-spacing"><b>Document Type</b></p>', unsafe_allow_html=True)
    with col3:
        st.markdown('<p class="row-spacing"><b>Date</b></p>', unsafe_allow_html=True)
    with col4:
        st.markdown('<p class="row-spacing"><b>Remarks</b></p>', unsafe_allow_html=True)
    
    # Add a separator line
    st.markdown("<hr>", unsafe_allow_html=True)
    
    # Add rows for each document
    for idx, doc in enumerate(documents):
        doc_id = f"General_{idx}"
        
        # Get metadata values, use 'N/A' if not available
        doc_type = doc.get('Doc_Type_General') or "N/A"
        date = doc.get('Date_General') or "N/A"
        remarks = doc.get('Remarks_General') or "N/A"
        
        # Create row using columns with row-spacing class
        col1, col2, col3, col4 = st.columns([0.5, 2, 1.5, 2.5])
        with col1:
            if st.button("👁️", key=f"view_{doc_id}"):
                st.session_state.selected_doc_id = doc_id
                st.session_state.viewing_document = True
                st.experimental_rerun()
        with col2:
            st.markdown(f'<span class="row-spacing">{doc_type}</span>', unsafe_allow_html=True)
        with col3:
            st.markdown(f'<span class="row-spacing">{date}</span>', unsafe_allow_html=True)
        with col4:
            st.markdown(f'<span class="row-spacing">{remarks}</span>', unsafe_allow_html=True)
        
        # Add a light separator between rows with reduced margins
        st.markdown('<hr style="margin: 1px 0; border: 0; border-top: 1px solid #eee;">', unsafe_allow_html=True)

def display_hr_library_table(documents):
    """Display HR library documents in table format with reduced spacing"""
    # Using Streamlit columns for the table header
    col1, col2, col3, col4, col5, col6 = st.columns([0.5, 1, 1.5, 1.5, 1.5, 1])
    with col1:
        st.markdown('<p class="row-spacing"><b>View</b></p>', unsafe_allow_html=True)
    with col2:
        st.markdown('<p class="row-spacing"><b>Emp #</b></p>', unsafe_allow_html=True)
    with col3:
        st.markdown('<p class="row-spacing"><b>Department</b></p>', unsafe_allow_html=True)
    with col4:
        st.markdown('<p class="row-spacing"><b>Document Type</b></p>', unsafe_allow_html=True)
    with col5:
        st.markdown('<p class="row-spacing"><b>Name</b></p>', unsafe_allow_html=True)
    with col6:
        st.markdown('<p class="row-spacing"><b>Date</b></p>', unsafe_allow_html=True)
    
    # Add a separator line
    st.markdown("<hr>", unsafe_allow_html=True)
    
    # Display each document as a row
    for idx, doc in enumerate(documents):
        doc_id = f"HR_{idx}"
        
        # Get metadata values
        emp_no = doc.get('Employee_No_HR') or "N/A"
        dept = doc.get('Department_HR') or "N/A"
        doc_type = doc.get('Document_Type_HR') or "N/A"
        name = doc.get('Name_HR') or "N/A"
        date = doc.get('Date_HR') or "N/A"
        
        # Create row using columns
        col1, col2, col3, col4, col5, col6 = st.columns([0.5, 1, 1.5, 1.5, 1.5, 1])
        with col1:
            if st.button("👁️", key=f"view_{doc_id}"):
                st.session_state.selected_doc_id = doc_id
                st.session_state.viewing_document = True
                st.experimental_rerun()
        with col2:
            st.markdown(f'<span class="row-spacing">{emp_no}</span>', unsafe_allow_html=True)
        with col3:
            st.markdown(f'<span class="row-spacing">{dept}</span>', unsafe_allow_html=True)
        with col4:
            st.markdown(f'<span class="row-spacing">{doc_type}</span>', unsafe_allow_html=True)
        with col5:
            st.markdown(f'<span class="row-spacing">{name}</span>', unsafe_allow_html=True)
        with col6:
            st.markdown(f'<span class="row-spacing">{date}</span>', unsafe_allow_html=True)
        
        # Add a light separator between rows with reduced margins
        st.markdown('<hr style="margin: 1px 0; border: 0; border-top: 1px solid #eee;">', unsafe_allow_html=True)

def display_florix_library_table(documents):
    """Display Florix library documents in table format with reduced spacing"""
    # Using Streamlit columns for the table header
    col1, col2, col3 = st.columns([0.5, 2, 3])
    with col1:
        st.markdown('<p class="row-spacing"><b>View</b></p>', unsafe_allow_html=True)
    with col2:
        st.markdown('<p class="row-spacing"><b>Document Type</b></p>', unsafe_allow_html=True)
    with col3:
        st.markdown('<p class="row-spacing"><b>Remarks</b></p>', unsafe_allow_html=True)
    
    # Add a separator line
    st.markdown("<hr>", unsafe_allow_html=True)
    
    # Display each document as a row
    for idx, doc in enumerate(documents):
        doc_id = f"Florix_{idx}"
        
        # Get metadata values
        doc_type = doc.get('Document_Type_Florix') or "N/A"
        remarks = doc.get('Remarks_Florix') or "N/A"
        
        # Create row using columns
        col1, col2, col3 = st.columns([0.5, 2, 3])
        with col1:
            if st.button("👁️", key=f"view_{doc_id}"):
                st.session_state.selected_doc_id = doc_id
                st.session_state.viewing_document = True
                st.experimental_rerun()
        with col2:
            st.markdown(f'<span class="row-spacing">{doc_type}</span>', unsafe_allow_html=True)
        with col3:
            st.markdown(f'<span class="row-spacing">{remarks}</span>', unsafe_allow_html=True)
        
        # Add a light separator between rows with reduced margins
        st.markdown('<hr style="margin: 1px 0; border: 0; border-top: 1px solid #eee;">', unsafe_allow_html=True)

def display_dftropio_library_table(documents):
    """Display DFTROPIO library documents in table format with reduced spacing"""
    # Create scrollable container
    st.markdown('<div style="overflow-x: auto;">', unsafe_allow_html=True)
    
    # Using Streamlit columns for the table header - adjust widths as needed
    cols = st.columns([0.4, 0.8, 1.2, 0.8, 1, 1.2, 0.8, 1, 0.8])
    
    with cols[0]:
        st.markdown('<p class="row-spacing"><b>View</b></p>', unsafe_allow_html=True)
    with cols[1]:
        st.markdown('<p class="row-spacing"><b>Serial No</b></p>', unsafe_allow_html=True)
    with cols[2]:
        st.markdown('<p class="row-spacing"><b>Name</b></p>', unsafe_allow_html=True)
    with cols[3]:
        st.markdown('<p class="row-spacing"><b>DOB</b></p>', unsafe_allow_html=True)
    with cols[4]:
        st.markdown('<p class="row-spacing"><b>Book Category</b></p>', unsafe_allow_html=True)
    with cols[5]:
        st.markdown('<p class="row-spacing"><b>Description</b></p>', unsafe_allow_html=True)
    with cols[6]:
        st.markdown('<p class="row-spacing"><b>Volume No</b></p>', unsafe_allow_html=True)
    with cols[7]:
        st.markdown('<p class="row-spacing"><b>Serial Range</b></p>', unsafe_allow_html=True)
    with cols[8]:
        st.markdown('<p class="row-spacing"><b>Act Number</b></p>', unsafe_allow_html=True)
    
    # Add a separator line
    st.markdown("<hr>", unsafe_allow_html=True)
    
    # Display each document as a row with compact styling
    for idx, doc in enumerate(documents):
        doc_id = f"DFTROPIO_{idx}"
        
        # Get metadata values with fallback to N/A
        serial_no = doc.get('SERIAL_NO_DFTROPIO') or "N/A"
        name = doc.get('Name_DFTROPIO') or "N/A"
        dob = doc.get('DOB_DFTROPIO') or "N/A"
        book_category = doc.get('BOOK_CATEGORY_DFTROPIO') or "N/A"
        description = doc.get('DESCRIPTION_DFTROPIO') or "N/A"
        volume_number = doc.get('VOLUME_NUMBER_DFTROPIO') or "N/A"
        serial_range = doc.get('SERIAL_RANGE_DFTROPIO') or "N/A"
        act_number = doc.get('ACT_NUMBER_DFTROPIO') or "N/A"
        
        # Create row with all fields using columns
        cols = st.columns([0.4, 0.8, 1.2, 0.8, 1, 1.2, 0.8, 1, 0.8])
        
        with cols[0]:
            if st.button("👁️", key=f"view_{doc_id}"):
                st.session_state.selected_doc_id = doc_id
                st.session_state.viewing_document = True
                st.experimental_rerun()
        
        # Apply compact styling to each cell
        with cols[1]:
            st.markdown(f'<div class="compact-text row-spacing">{serial_no}</div>', unsafe_allow_html=True)
        with cols[2]:
            st.markdown(f'<div class="compact-text row-spacing">{name}</div>', unsafe_allow_html=True)
        with cols[3]:
            st.markdown(f'<div class="compact-text row-spacing">{dob}</div>', unsafe_allow_html=True)
        with cols[4]:
            st.markdown(f'<div class="compact-text row-spacing">{book_category}</div>', unsafe_allow_html=True)
        with cols[5]:
            st.markdown(f'<div class="compact-text row-spacing">{description}</div>', unsafe_allow_html=True)
        with cols[6]:
            st.markdown(f'<div class="compact-text row-spacing">{volume_number}</div>', unsafe_allow_html=True)
        with cols[7]:
            st.markdown(f'<div class="compact-text row-spacing">{serial_range}</div>', unsafe_allow_html=True)
        with cols[8]:
            st.markdown(f'<div class="compact-text row-spacing">{act_number}</div>', unsafe_allow_html=True)
        
        # Add a light separator between rows with reduced margins
        st.markdown('<hr style="margin: 1px 0; border: 0; border-top: 1px solid #eee;">', unsafe_allow_html=True)
    
    # Close the scrollable container
    st.markdown('</div>', unsafe_allow_html=True)

def display_finance_library_table(documents):
    """Display Finance library documents in table format with reduced spacing"""
    # Using Streamlit columns for the table header
    col1, col2, col3, col4, col5 = st.columns([0.5, 1.5, 1.5, 1, 2])
    with col1:
        st.markdown('<p class="row-spacing"><b>View</b></p>', unsafe_allow_html=True)
    with col2:
        st.markdown('<p class="row-spacing"><b>Document ID</b></p>', unsafe_allow_html=True)
    with col3:
        st.markdown('<p class="row-spacing"><b>Document Type</b></p>', unsafe_allow_html=True)
    with col4:
        st.markdown('<p class="row-spacing"><b>Date</b></p>', unsafe_allow_html=True)
    with col5:
        st.markdown('<p class="row-spacing"><b>Info</b></p>', unsafe_allow_html=True)
    
    # Add a separator line
    st.markdown("<hr>", unsafe_allow_html=True)
    
    # Display each document as a row
    for idx, doc in enumerate(documents):
        doc_id = f"Finance_{idx}"
        
        # Get metadata values
        document_id = doc.get('Document_ID_Finance') or "N/A"
        doc_type = doc.get('Document_Type_Finance') or "N/A"
        date = doc.get('Date_Finance') or "N/A"
        info = doc.get('Info_Finance') or "N/A"
        
        # Create row using columns
        col1, col2, col3, col4, col5 = st.columns([0.5, 1.5, 1.5, 1, 2])
        with col1:
            if st.button("👁️", key=f"view_{doc_id}"):
                st.session_state.selected_doc_id = doc_id
                st.session_state.viewing_document = True
                st.experimental_rerun()
        with col2:
            st.markdown(f'<span class="row-spacing">{document_id}</span>', unsafe_allow_html=True)
        with col3:
            st.markdown(f'<span class="row-spacing">{doc_type}</span>', unsafe_allow_html=True)
        with col4:
            st.markdown(f'<span class="row-spacing">{date}</span>', unsafe_allow_html=True)
        with col5:
            st.markdown(f'<span class="row-spacing">{info}</span>', unsafe_allow_html=True)
        
        # Add a light separator between rows with reduced margins
        st.markdown('<hr style="margin: 1px 0; border: 0; border-top: 1px solid #eee;">', unsafe_allow_html=True)

def display_ayala_annual_report_library_table(documents):
    """Display Ayala Annual Report library documents in table format with reduced spacing"""
    # Using Streamlit columns for the table header
    col1, col2, col3, col4, col5 = st.columns([0.5, 2, 1, 1.5, 2])
    with col1:
        st.markdown('<p class="row-spacing"><b>View</b></p>', unsafe_allow_html=True)
    with col2:
        st.markdown('<p class="row-spacing"><b>Name</b></p>', unsafe_allow_html=True)
    with col3:
        st.markdown('<p class="row-spacing"><b>Year</b></p>', unsafe_allow_html=True)
    with col4:
        st.markdown('<p class="row-spacing"><b>Document Type</b></p>', unsafe_allow_html=True)
    with col5:
        st.markdown('<p class="row-spacing"><b>Remarks</b></p>', unsafe_allow_html=True)
    
    # Add a separator line
    st.markdown("<hr>", unsafe_allow_html=True)
    
    # Display each document as a row
    for idx, doc in enumerate(documents):
        doc_id = f"Ayala_Annual_Report_{idx}"
        
        # Debug: Print what's actually in the document
        print(f"DEBUG - Document keys: {list(doc.keys())}")
        print(f"DEBUG - Ayala fields: Name={doc.get('Name_Ayala_Annual_Report')}, Year={doc.get('Year_Ayala_Annual_Report')}")

        # Get metadata values
        name = doc.get('Name_Ayala_Annual_Report') or "N/A"
        year = doc.get('Year_Ayala_Annual_Report') or "N/A"
        doc_type = doc.get('DocumentType_Ayala_Annual_Report') or "N/A"
        remarks = doc.get('Remarks_Ayala_Annual_Report') or "N/A"
        
        # Create row using columns
        col1, col2, col3, col4, col5 = st.columns([0.5, 2, 1, 1.5, 2])
        with col1:
            if st.button("👁️", key=f"view_{doc_id}"):
                st.session_state.selected_doc_id = doc_id
                st.session_state.viewing_document = True
                st.experimental_rerun()
        with col2:
            st.markdown(f'<span class="row-spacing">{name}</span>', unsafe_allow_html=True)
        with col3:
            st.markdown(f'<span class="row-spacing">{year}</span>', unsafe_allow_html=True)
        with col4:
            st.markdown(f'<span class="row-spacing">{doc_type}</span>', unsafe_allow_html=True)
        with col5:
            st.markdown(f'<span class="row-spacing">{remarks}</span>', unsafe_allow_html=True)
        
        # Add a light separator between rows with reduced margins
        st.markdown('<hr style="margin: 1px 0; border: 0; border-top: 1px solid #eee;">', unsafe_allow_html=True)

def display_ayala_legal_docs_library_table(documents):
    """Display Ayala Legal Docs library documents in table format with reduced spacing"""
    # Using Streamlit columns for the table header
    col1, col2, col3, col4 = st.columns([0.5, 2.5, 1.5, 2.5])
    with col1:
        st.markdown('<p class="row-spacing"><b>View</b></p>', unsafe_allow_html=True)
    with col2:
        st.markdown('<p class="row-spacing"><b>Name</b></p>', unsafe_allow_html=True)
    with col3:
        st.markdown('<p class="row-spacing"><b>Document Type</b></p>', unsafe_allow_html=True)
    with col4:
        st.markdown('<p class="row-spacing"><b>Remarks</b></p>', unsafe_allow_html=True)
    
    # Add a separator line
    st.markdown("<hr>", unsafe_allow_html=True)
    
    # Display each document as a row
    for idx, doc in enumerate(documents):
        doc_id = f"Ayala_Legal_Docs_{idx}"
        
        # Get metadata values
        name = doc.get('Name_Ayala_Legal_Docs') or "N/A"
        doc_type = doc.get('DocumentType_Ayala_Legal_Docs') or "N/A"
        remarks = doc.get('Remarks_Ayala_Legal_Docs') or "N/A"
        
        # Create row using columns
        col1, col2, col3, col4 = st.columns([0.5, 2.5, 1.5, 2.5])
        with col1:
            if st.button("👁️", key=f"view_{doc_id}"):
                st.session_state.selected_doc_id = doc_id
                st.session_state.viewing_document = True
                st.experimental_rerun()
        with col2:
            st.markdown(f'<span class="row-spacing">{name}</span>', unsafe_allow_html=True)
        with col3:
            st.markdown(f'<span class="row-spacing">{doc_type}</span>', unsafe_allow_html=True)
        with col4:
            st.markdown(f'<span class="row-spacing">{remarks}</span>', unsafe_allow_html=True)
        
        # Add a light separator between rows with reduced margins
        st.markdown('<hr style="margin: 1px 0; border: 0; border-top: 1px solid #eee;">', unsafe_allow_html=True)

def display_library_documents(library, documents):
    """Display documents for a specific library with the appropriate table format"""
    # Choose the appropriate display function based on the library
    if library == "General":
        display_general_library_table(documents)
    elif library == "HR":
        display_hr_library_table(documents)
    elif library == "Florix":
        display_florix_library_table(documents)
    elif library == "DFTROPIO":
        display_dftropio_library_table(documents)
    elif library == "Finance":
        display_finance_library_table(documents)
    elif library == "Ayala_Annual_Report":
        display_ayala_annual_report_library_table(documents)
    elif library == "Ayala_Legal_Docs":
        display_ayala_legal_docs_library_table(documents)
    else:
        # Default display for unknown libraries
        st.warning(f"No custom display format for library: {library}")
        for idx, doc in enumerate(documents):
            doc_name = doc.get('DocumentName', f'Document {idx+1}')
            doc_id = f"{library}_{idx}"
            if st.button(f"📄 {doc_name}", key=f"doc_{doc_id}", use_container_width=True):
                st.session_state.selected_doc_id = doc_id
                st.session_state.viewing_document = True
                st.experimental_rerun()

def display_similar_documents():
    """Display similar documents grouped by library"""
    # Use h3 for smaller title
    st.markdown("<h3>Similar Documents</h3>", unsafe_allow_html=True)
    
    if not st.session_state.similar_docs:
        st.info("No similar documents found with the selected entities.")
        if st.button("← Back to Search"):
            st.session_state.show_similar_docs = False
            st.session_state.similar_doc_history = []
            st.experimental_rerun()
        return
    
    # Add navigation buttons
    col1, col2, col3 = st.columns([1, 1, 3])
    with col1:
        if st.button("← Back to Search"):
            st.session_state.show_similar_docs = False
            st.session_state.similar_doc_history = []
            st.session_state.current_doc_content = None
            st.experimental_rerun()
    
    with col2:
        if len(st.session_state.similar_doc_history) > 1:
            if st.button("← Previous Results"):
                # Restore previous state
                st.session_state.similar_doc_history.pop()  # Remove current
                previous_state = st.session_state.similar_doc_history[-1]  # Get previous
                
                # Restore entity selections
                st.session_state.selected_people = set(previous_state['entities']['people'])
                st.session_state.selected_organizations = set(previous_state['entities']['organizations'])
                st.session_state.selected_locations = set(previous_state['entities']['locations'])
                
                # Reset current document content
                st.session_state.current_doc_content = None
                
                # Get documents for previous state
                related_docs = get_related_documents(
                    st.session_state.gremlin_client,
                    st.session_state.selected_people,
                    st.session_state.selected_organizations,
                    st.session_state.selected_locations
                )
                
                if related_docs:
                    st.session_state.similar_docs = related_docs
                st.experimental_rerun()
    
    # Display current search path
    st.write("### Search Path")
    for idx, history_item in enumerate(st.session_state.similar_doc_history, 1):
        entities = history_item['entities']
        st.write(f"**Search {idx}:** ", end="")
        entity_parts = []
        if entities['people']:
            entity_parts.append(f"People: {', '.join(entities['people'])}")
        if entities['organizations']:
            entity_parts.append(f"Organizations: {', '.join(entities['organizations'])}")
        if entities['locations']:
            entity_parts.append(f"Locations: {', '.join(entities['locations'])}")
        st.write(" | ".join(entity_parts))
    
    st.write("### Similar Documents Found")
    
    # Group documents by library
    library_groups = {}
    for doc in st.session_state.similar_docs:
        library = doc['library']
        if library not in library_groups:
            library_groups[library] = []
        library_groups[library].append(doc)
    
    # Display documents grouped by library
    for library, documents in library_groups.items():
        with st.expander(f"{library} ({len(documents)} documents)", expanded=False):
            for doc in documents:
                doc_name = doc['document']
                matched_entities = doc['matched_entities']
                
                # Create a container for each document
                doc_container = st.container()
                with doc_container:
                    # Display document name and button
                    col1, col2 = st.columns([4, 1])
                    with col1:
                        st.markdown(f'<span class="row-spacing">📄 {doc_name}</span>', unsafe_allow_html=True)
                    with col2:
                        if st.button("View Document", 
                                   key=f"view_similar_{get_hash(doc_name)}", 
                                   use_container_width=True):
                            # Fetch full document from Azure Search
                            results = search_documents(
                                st.session_state.search_client, 
                                f"DocumentName eq '{doc_name}'"
                            )
                            if results:
                                st.session_state.current_doc_content = results[0]
                                st.session_state.viewing_document = True  # Set viewing document state
                                st.experimental_rerun()
                    
                    # Display matched entities
                    st.markdown("**Entities in this Document:**")
                    matched_selected = {
                        'People': [],
                        'Organizations': [],
                        'Locations': []
                    }
                    
                    # Check which selected entities are present in this document
                    # Using 'peopl' to match your database label
                    if 'peopl' in matched_entities:
                        for person in matched_entities['peopl']:
                            if person in st.session_state.selected_people:
                                matched_selected['People'].append(f"<span style='color: #FF6B6B'>{person}</span>")
                            else:
                                matched_selected['People'].append(person)
                    
                    if 'organization' in matched_entities:
                        for org in matched_entities['organization']:
                            if org in st.session_state.selected_organizations:
                                matched_selected['Organizations'].append(f"<span style='color: #FF6B6B'>{org}</span>")
                            else:
                                matched_selected['Organizations'].append(org)
                    
                    if 'location' in matched_entities:
                        for location in matched_entities['location']:
                            if location in st.session_state.selected_locations:
                                matched_selected['Locations'].append(f"<span style='color: #FF6B6B'>{location}</span>")
                            else:
                                matched_selected['Locations'].append(location)
                    
                    # Display entities with selected ones highlighted
                    entities_container = st.container()
                    with entities_container:
                        if matched_selected['People']:
                            st.markdown("**People:**", unsafe_allow_html=True)
                            st.markdown(", ".join(matched_selected['People']), unsafe_allow_html=True)
                        
                        if matched_selected['Organizations']:
                            st.markdown("**Organizations:**", unsafe_allow_html=True)
                            st.markdown(", ".join(matched_selected['Organizations']), unsafe_allow_html=True)
                        
                        if matched_selected['Locations']:
                            st.markdown("**Locations:**", unsafe_allow_html=True)
                            st.markdown(", ".join(matched_selected['Locations']), unsafe_allow_html=True)
                    
                    # Add a note about colored entities
                    if (any('FF6B6B' in item for sublist in matched_selected.values() for item in sublist)):
                        st.markdown("<span style='color: #FF6B6B'>▲</span> Highlighted entities are from your selection", unsafe_allow_html=True)
                    
                    st.markdown('<hr style="margin: 5px 0;">', unsafe_allow_html=True)

def main():
    # Initialize session state
    init_session_state()
    
    # Apply table styles
    apply_table_styles()
    
    # Display header on every screen
    display_header()
    
    if st.session_state.viewing_document:
        # Show back button
        back_button_label = "← Back to Search" if not st.session_state.show_similar_docs else "← Back to Similar Documents"
        if st.button(back_button_label):
            st.session_state.viewing_document = False
            st.experimental_rerun()
        
        # Display current document content
        if st.session_state.show_similar_docs and st.session_state.current_doc_content:
            display_document_content(
                st.session_state.current_doc_content,
                get_hash(st.session_state.current_doc_content.get('DocumentName')),
                is_similar_view=True
            )
        elif st.session_state.selected_doc_id:
            # Get library and index from selected_doc_id
            doc_id_parts = st.session_state.selected_doc_id.split('_')
            if len(doc_id_parts) >= 2:
                library = '_'.join(doc_id_parts[:-1])  # Join all parts except the last one
                idx_str = doc_id_parts[-1]  # Last part is the index
            else:
                library = doc_id_parts[0]
                idx_str = "0"
            idx = int(idx_str)
            
            # Get all documents for this library
            library_docs = []
            for lib, docs in group_by_library(st.session_state.search_results).items():
                if lib == library:
                    library_docs = docs
                    break
            
            if idx < len(library_docs):
                display_document_content(library_docs[idx], st.session_state.selected_doc_id)
    elif st.session_state.show_similar_docs:
        display_similar_documents()
    else:
        # Reset similar document history when starting new search
        st.session_state.similar_doc_history = []
        st.session_state.current_doc_content = None
        
        # Search input
        search_query = st.text_input("Enter your search query")
        
        if search_query:
            # Perform search and store results in session state
            st.session_state.search_results = search_documents(st.session_state.search_client, search_query)
        
        # Display results if we have them
        if st.session_state.search_results:
            # Group results by library
            grouped_results = group_by_library(st.session_state.search_results)
            
            # Display count of results
            total_results = sum(len(docs) for docs in grouped_results.values())
            st.write(f"Found {total_results} documents across {len(grouped_results)} libraries")
            
            # Display results grouped by library
            for library, documents in grouped_results.items():
                with st.expander(f"{library} ({len(documents)} documents)", expanded=False):
                    # Create a container for the library's documents
                    doc_container = st.container()
                    with doc_container:
                        # Display the documents in the appropriate format for this library
                        display_library_documents(library, documents)

if __name__ == "__main__":
    main()
