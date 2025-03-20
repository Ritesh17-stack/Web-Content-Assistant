import streamlit as st
import requests
from bs4 import BeautifulSoup
import pandas as pd
from langchain_groq import ChatGroq
from langchain.schema import SystemMessage, HumanMessage
import os
from dotenv import load_dotenv
import re
from urllib.parse import urljoin

# Load environment variables from .env file
load_dotenv()

# Set up the page
st.set_page_config(page_title="Web Content Assistant", layout="wide")
st.title("Web Content Assistant")

# Get API key from environment variables
api_key = os.getenv("GROQ_API_KEY")
if not api_key:
    st.sidebar.warning("⚠️ No GROQ_API_KEY found in environment variables. LLM features will not work.")
else:
    st.sidebar.success("✅ GROQ_API_KEY loaded from environment variables")

model_name = st.sidebar.selectbox("Select Groq Model", ["llama3-8b-8192", "mixtral-8x7b-32768", "gemma-7b-it"])
def scrape_website(url):
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
            'Cache-Control': 'max-age=0'
        }
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Extract title
        title = soup.title.string if soup.title else "No title found"
        
        # Extract metadata
        meta_data = {}
        for meta in soup.find_all('meta'):
            if meta.get('name'):
                meta_data[meta.get('name')] = meta.get('content')
            elif meta.get('property'):
                meta_data[meta.get('property')] = meta.get('content')
        
        description = meta_data.get('description', "No description found")
        
        # Extract all visible text content (more comprehensive approach)
        # First remove script and style elements
        for script_or_style in soup(['script', 'style', 'noscript', 'iframe', 'head']):
            script_or_style.extract()
        
        # Get all text
        all_text = soup.get_text(separator=' ', strip=True)
        # Clean up whitespace
        all_text = re.sub(r'\s+', ' ', all_text).strip()
        
        # Extract structured content with hierarchy
        structured_content = {}
        
        # Headers
        headers = {}
        for i in range(1, 7):
            headers[f'h{i}'] = [h.get_text(strip=True) for h in soup.find_all(f'h{i}')]
        
        # Main content blocks
        content_blocks = []
        for tag in ['article', 'section', 'div', 'main']:
            main_elements = soup.find_all(tag, class_=lambda x: x and any(term in str(x).lower() for term in 
                                                                       ['content', 'main', 'article', 'body', 'text']))
            for element in main_elements:
                block_text = element.get_text(separator=' ', strip=True)
                if len(block_text) > 100:  # Only substantial blocks
                    content_blocks.append({
                        'type': tag,
                        'text': re.sub(r'\s+', ' ', block_text)
                    })
        
        # Paragraphs
        paragraphs = []
        for p in soup.find_all('p'):
            text = p.get_text(strip=True)
            if len(text) > 10:  # Ignore very short paragraphs
                paragraphs.append(text)
        
        # Lists
        lists = []
        for list_tag in soup.find_all(['ul', 'ol']):
            items = [li.get_text(strip=True) for li in list_tag.find_all('li')]
            if items:
                lists.append({
                    'type': list_tag.name,
                    'items': items
                })
        
        # Tables
        tables = []
        for table in soup.find_all('table'):
            table_data = []
            rows = table.find_all('tr')
            
            # Try to find headers
            headers_list = []
            th_elements = table.find_all('th')
            if th_elements:
                headers_list = [th.get_text(strip=True) for th in th_elements]
            
            # Process rows
            for row in rows:
                cells = row.find_all(['td', 'th'])
                if cells:
                    row_data = [cell.get_text(strip=True) for cell in cells]
                    table_data.append(row_data)
            
            if table_data:
                tables.append({
                    'headers': headers_list,
                    'data': table_data
                })
        
        # Images
        images = []
        for img in soup.find_all('img'):
            src = img.get('src', '')
            alt = img.get('alt', 'No description')
            if src:
                # Convert relative URLs to absolute
                if not src.startswith(('http://', 'https://')):
                    src = urljoin(url, src)
                images.append({
                    'src': src,
                    'alt': alt
                })
        
        # Links
        links = []
        for a in soup.find_all('a', href=True):
            href = a['href']
            text = a.get_text(strip=True)
            # Convert relative URLs to absolute
            if not href.startswith(('http://', 'https://', 'mailto:', 'tel:')):
                href = urljoin(url, href)
            links.append({
                'url': href,
                'text': text if text else href
            })
        
        # Combine structured content
        structured_content = {
            'headers': headers,
            'content_blocks': content_blocks,
            'paragraphs': paragraphs,
            'lists': lists,
            'tables': tables,
            'images': images[:30]  # Limit number of images
        }
    
        # Prioritize content blocks and paragraphs for the combined content
        combined_content = ""
        
        # Add headers as sections
        for level in range(1, 7):
            for header in headers.get(f'h{level}', []):
                combined_content += f"\n{'#' * level} {header}\n"
        
        # Add main content blocks
        for block in content_blocks:
            combined_content += f"\n{block['text']}\n"
        
        # Add paragraphs if content is still small
        if len(combined_content) < 3000:
            for p in paragraphs:
                combined_content += f"\n{p}\n"
        
        # Add lists
        for lst in lists:
            combined_content += "\n"
            for i, item in enumerate(lst['items']):
                prefix = "- " if lst['type'] == 'ul' else f"{i+1}. "
                combined_content += f"{prefix}{item}\n"
        
        # If still not enough content, use all_text
        if len(combined_content.strip()) < 1000:
            combined_content = all_text
        
        return {
            "title": title,
            "description": description,
            "meta_data": meta_data,
            "all_text": all_text,
            "structured_content": structured_content,
            "combined_content": combined_content,
            "links": links[:50]  # Limit to first 50 links
        }
    
    except Exception as e:
        return {"error": str(e)}

# Summarize function
def summarize_content(scraped_data):
    if not api_key:
        return "Error: No GROQ_API_KEY found in environment variables. Please add it to your .env file."
    
    try:
        llm = ChatGroq(
            api_key=api_key,
            model_name=model_name
        )
        
        system_prompt = "You are a helpful assistant that summarizes web content concisely."
        user_prompt = f"Summarize this web content in 3-5 paragraphs:\n\nTitle: {scraped_data['title']}\n\n{scraped_data['combined_content'][:7500]}"
        
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt)
        ]
        
        response = llm.invoke(messages)
        return response.content
    
    except Exception as e:
        return f"Error summarizing content: {str(e)}"

# Chat function
def chat_with_content(scraped_data, user_question):
    if not api_key:
        return "Error: No GROQ_API_KEY found in environment variables. Please add it to your .env file."
    
    try:
        llm = ChatGroq(
            api_key=api_key,
            model_name=model_name
        )
        
        system_prompt = f"""You are a helpful assistant that answers questions based on the web content provided.
The content is from the page titled: {scraped_data['title']}
When answering, only use information from the content provided. If the answer is not in the content, say that you don't have that information."""
        
        user_prompt = f"Content:\n{scraped_data['combined_content'][:7500]}\n\nQuestion: {user_question}"
        
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt)
        ]
        
        response = llm.invoke(messages)
        return response.content
    
    except Exception as e:
        return f"Error processing chat: {str(e)}"

# Initialize session state
if 'scraped_data' not in st.session_state:
    st.session_state.scraped_data = None

if 'summary' not in st.session_state:
    st.session_state.summary = None

if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

if 'new_message' not in st.session_state:
    st.session_state.new_message = False

# Function to handle new message and set flag
def handle_new_message(user_question):
    with st.spinner("Processing..."):
        answer = chat_with_content(st.session_state.scraped_data, user_question)
        st.session_state.chat_history.append((user_question, answer))
        st.session_state.new_message = True

# UI components for scraping
with st.sidebar:
    st.header("Scraper Settings")
    url_input = st.text_input("Enter URL to scrape")
    scrape_button = st.button("Scrape Website")
    
    if scrape_button and url_input:
        with st.spinner("Scraping website..."):
            st.session_state.scraped_data = scrape_website(url_input)
            # Reset chat history when scraping a new site
            st.session_state.chat_history = []
            
            if st.session_state.scraped_data and "error" not in st.session_state.scraped_data:
                with st.spinner("Generating summary..."):
                    st.session_state.summary = summarize_content(st.session_state.scraped_data)

# Main content area - tabs for different features
tab1, tab2 = st.tabs(["Website Content", "Chat with Content"])

# Tab 1: Website Content and Summary
with tab1:
    if st.session_state.scraped_data:
        if "error" in st.session_state.scraped_data:
            st.error(f"Error: {st.session_state.scraped_data['error']}")
        else:
            col1, col2 = st.columns(2)
            
            # Left column: Original content
            with col1:
                st.header("Scraped Content")
                st.subheader(st.session_state.scraped_data["title"])
                st.write("**Description:** " + st.session_state.scraped_data["description"])
                
                # Display structured content in expandable sections
                with st.expander("Headers", expanded=False):
                    for level, headers in st.session_state.scraped_data["structured_content"]["headers"].items():
                        if headers:
                            st.write(f"**{level.upper()}:**")
                            for header in headers:
                                st.write(f"- {header}")
                
                with st.expander("Main Content Blocks", expanded=False):
                    for i, block in enumerate(st.session_state.scraped_data["structured_content"]["content_blocks"]):
                        st.write(f"**Block {i+1} ({block['type']}):**")
                        st.write(block['text'][:500] + ("..." if len(block['text']) > 500 else ""))
                        st.write("---")
                
                with st.expander("Lists", expanded=False):
                    for i, lst in enumerate(st.session_state.scraped_data["structured_content"]["lists"]):
                        st.write(f"**List {i+1} ({lst['type']}):**")
                        for item in lst['items']:
                            st.write(f"- {item}")
                        st.write("---")
                
                with st.expander("Tables", expanded=False):
                    for i, table in enumerate(st.session_state.scraped_data["structured_content"]["tables"]):
                        st.write(f"**Table {i+1}:**")
                        if table['headers']:
                            st.write("Headers: " + ", ".join(table['headers']))
                        
                        # Convert to pandas DataFrame for display
                        if table['data']:
                            df = pd.DataFrame(table['data'])
                            if table['headers'] and len(table['headers']) == df.shape[1]:
                                df.columns = table['headers']
                            st.dataframe(df)
                        st.write("---")
                
                with st.expander("Images", expanded=False):
                    for i, img in enumerate(st.session_state.scraped_data["structured_content"]["images"]):
                        st.write(f"**Image {i+1}:**")
                        st.write(f"Alt text: {img['alt']}")
                        st.write(f"Source: {img['src']}")
                        st.write("---")
                
                with st.expander("Links", expanded=False):
                    for i, link in enumerate(st.session_state.scraped_data["links"]):
                        st.write(f"[{link['text']}]({link['url']})")
                
                with st.expander("Raw Combined Content", expanded=False):
                    st.text_area("Content", st.session_state.scraped_data["combined_content"][:10000], height=300)
                
                with st.expander("Metadata", expanded=False):
                    for key, value in st.session_state.scraped_data["meta_data"].items():
                        st.write(f"**{key}:** {value}")
            
            # Right column: Summary
            with col2:
                st.header("Summary")
                if st.session_state.summary:
                    st.markdown(st.session_state.summary)
                else:
                    st.info("Summary will appear here after scraping.")
    else:
        st.info("Enter a URL in the sidebar and click 'Scrape Website' to get started.")

# Tab 2: Chat with content
with tab2:
    if st.session_state.scraped_data and "error" not in st.session_state.scraped_data:
        st.header(f"Chat about: {st.session_state.scraped_data['title']}")
        
        # Display chat history
        chat_container = st.container()
        with chat_container:
            for i, (question, answer) in enumerate(st.session_state.chat_history):
                st.markdown(f"**You:** {question}")
                st.markdown(f"**Assistant:** {answer}")
                st.markdown("---")
        
        # Chat input
        user_question = st.text_input("Ask a question about this content:")
        send_button = st.button("Send")
        
        if send_button and user_question:
            handle_new_message(user_question)
            st.rerun()  # Use st.rerun() instead of experimental_rerun
    else:
        st.info("Scrape a website first to chat with its content.")

# Reset new message flag after rerun
if st.session_state.new_message:
    st.session_state.new_message = False

# Add some helpful instructions
st.sidebar.markdown("---")
st.sidebar.markdown("""
### How to use:
1. Create a `.env` file in the same directory with your Groq API key:
   ```
   GROQ_API_KEY=your_api_key_here
   ```
2. Enter a URL to scrape in the sidebar
3. Click 'Scrape Website' to extract and summarize the content
4. View the content and summary in the first tab
5. Ask questions about the content in the Chat tab
""")