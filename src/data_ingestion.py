import urllib.request
import xml.etree.ElementTree as ET
import logging
import json
import re
import os
import pandas as pd
from pathlib import Path
from typing import List, Dict, Optional

# Configure logging for the pipeline
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class ArxivDataPipeline:
    def __init__(self, query: str = "all:electron", max_results: int = 100000):
        self.base_url = "http://export.arxiv.org/api/query"
        self.query = query
        self.max_results = max_results
        self.namespace = {'atom': 'http://www.w3.org/2005/Atom'}

    def fetch_data(self, start: int = 0, batch_size: int = 1000) -> Optional[str]:
        url = f"{self.base_url}?search_query={self.query}&start={start}&max_results={batch_size}"
        logging.info(f"Fetching data from {url}")
        try:
            import ssl
            context = ssl._create_unverified_context()
            with urllib.request.urlopen(url, context=context) as response:
                if response.status == 200:
                    return response.read().decode('utf-8')
                else:
                    logging.error(f"Failed to fetch data. HTTP Status: {response.status}")
                    return None
        except Exception as e:
            logging.error(f"Error during API request: {str(e)}")
            return None

    def clean_text(self, text: str) -> str:
        if not text:
            return ""
        # Remove consecutive whitespaces, newlines, and tabs
        text = re.sub(r'\s+', ' ', text)
        # Strip leading and trailing spaces
        return text.strip()

    def parse_and_clean(self, xml_data: str) -> List[Dict[str, str]]:
        logging.info("Parsing XML and cleaning data...")
        try:
            root = ET.fromstring(xml_data)
        except ET.ParseError as e:
            logging.error(f"Failed to parse XML: {str(e)}")
            return []

        papers = []
        entries = root.findall('atom:entry', self.namespace)
        
        for entry in entries:
            title_node = entry.find('atom:title', self.namespace)
            summary_node = entry.find('atom:summary', self.namespace)
            id_node = entry.find('atom:id', self.namespace)
            category_node = entry.find('atom:category', self.namespace)

            # Extract raw text, handling missing nodes safely
            raw_title = title_node.text if title_node is not None else "Unknown Title"
            raw_summary = summary_node.text if summary_node is not None else ""
            paper_id = id_node.text if id_node is not None else "Unknown ID"
            
            # Extract category
            category = category_node.attrib.get('term') if category_node is not None else "Unknown Category"

            # Clean the extracted text
            clean_title = self.clean_text(raw_title)
            clean_summary = self.clean_text(raw_summary)

            if not clean_summary:
                logging.warning(f"Paper {paper_id} has an empty abstract. Skipping.")
                continue

            papers.append({
                "id": paper_id,
                "title": clean_title,
                "abstract": clean_summary,
                "category": category
            })

        logging.info(f"Successfully processed {len(papers)} papers.")
        return papers
        
    def save_data(self, data: List[Dict[str, str]], filepath: Path) -> None:
        logging.info(f"Saving data to {filepath}")
        filepath.parent.mkdir(parents=True, exist_ok=True)
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=4, ensure_ascii=False)
            logging.info("Data saved successfully.")
        except Exception as e:
            logging.error(f"Failed to save data: {e}")

    def run(self, save_path: Optional[Path] = None) -> List[Dict[str, str]]:
        import time
        all_papers = []
        batch_size = min(1000, self.max_results)
        
        for start_idx in range(0, self.max_results, batch_size):
            remaining = self.max_results - start_idx
            current_batch = min(batch_size, remaining)
            
            logging.info(f"Fetching batch {start_idx} to {start_idx + current_batch}...")
            raw_xml = self.fetch_data(start=start_idx, batch_size=current_batch)
            if raw_xml:
                papers = self.parse_and_clean(raw_xml)
                all_papers.extend(papers)
                
            # ArXiv API terms require 3 seconds between requests
            if start_idx + current_batch < self.max_results:
                time.sleep(3)
                
        if save_path and all_papers:
            self.save_data(all_papers, save_path)
            
        return all_papers

if __name__ == "__main__":
    # Industry benchmark: 10,000 to 50,000 abstracts for a solid base model.
    # We use a polite paginated loop, so 10,000 is perfectly safe without getting IP banned.
    pipeline = ArxivDataPipeline(query="all:quantum", max_results=10000)
    
    # Define our raw data path, locating the project root from this file's position
    project_root = Path(__file__).resolve().parent.parent
    raw_data_path = project_root / "data" / "raw" / "raw_papers.json"
    
    processed_data = pipeline.run(save_path=raw_data_path)
    
    # Save as CSV per user request
    os.makedirs("data/raw", exist_ok=True)
    df = pd.DataFrame(processed_data)
    df.to_csv("data/raw/arxiv_data.csv", index=False)
    print("Saved raw data to data/raw/arxiv_data.csv")
    
    for i, paper in enumerate(processed_data[:3]):
        print(f"--- Paper {i+1} ---")
        print(f"Title: {paper['title']}")
        print(f"Category: {paper['category']}")
        print(f"Abstract Preview: {paper['abstract'][:100]}...\n")
