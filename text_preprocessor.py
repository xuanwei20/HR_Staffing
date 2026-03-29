import pandas as pd
import nltk
import re
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer, PorterStemmer
from nltk.corpus import stopwords


class TextPreprocessor:
    """Handles text preprocessing with fallback mechanisms"""
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))
        self.stemmer = PorterStemmer()
        
        # Initialize lemmatizer, fall back to stemming if not available
        self.lemmatizer = None
        self.has_wordnet = False
        
        try:
            nltk.data.find('corpora/wordnet')
            self.lemmatizer = WordNetLemmatizer()
            self.has_wordnet = True
            print("WordNet lemmatizer initialized")
        except LookupError:
            print("WordNet not available, using stemming only")
        
        self.abbreviations = {
            'HR': 'human resources',
            'CHRO': 'chief human resources officer',
            'HRBP': 'human resources business partner',
            'HRIS': 'human resources information system',
            'SVP': 'senior vice president',
            'VP': 'vice president',
            'CEO': 'chief executive officer',
            'CFO': 'chief financial officer',
            'CTO': 'chief technology officer',
            'COO': 'chief operating officer',
            'DIR': 'director',
            'MGR': 'manager',
            'SR': 'senior',
            'JR': 'junior',
            'TECH': 'technology',
            'ENG': 'engineer',
            'DEV': 'developer',
            'SPEC': 'specialist',
            'ASSOC': 'associate',
            'ASST': 'assistant',
            'REC': 'recruiter',
            'TAL': 'talent',
            'ACQ': 'acquisition',
            'L&D': 'learning and development',
            'OD': 'organization development',
            'TA': 'talent acquisition'
        }

        self.location_mappings = {
            # Country mappings (Turkish to English)
            'Kanada': 'Canada',
            'Ä°zmir': 'Izmir',
            'TÃ¼rkiye': 'Turkey',
            'Ä°zmir, TÃ¼rkiye': 'Izmir, Turkey',
            'Amerika BirleÅŸik Devletleri': 'United States',
            'Amerika Birleşik Devletleri': 'United States',
            
            # Standardize area formats
            'Greater New York City Area': 'New York, NY',
            'New York, New York': 'New York, NY',
            'Greater Philadelphia Area': 'Philadelphia, PA',
            'Greater Boston Area': 'Boston, MA',
            'San Francisco Bay Area': 'San Francisco, CA',
            'Greater Los Angeles Area': 'Los Angeles, CA',
            'Greater Chicago Area': 'Chicago, IL',
            'Greater Atlanta Area': 'Atlanta, GA',
            'Dallas/Fort Worth Area': 'Dallas, TX',
            'Houston, Texas Area': 'Houston, TX',
            'Austin, Texas Area': 'Austin, TX',
            
            # Clean up city/state formatting
            'Raleigh-Durham, North Carolina Area': 'Raleigh, NC',
            'Raleigh-Durham, North Carolina': 'Raleigh, NC',
            'Denton, Texas': 'Denton, TX',
            'Lake Forest, California': 'Lake Forest, CA',
            'Atlanta, Georgia': 'Atlanta, GA',
            'Chicago, Illinois': 'Chicago, IL',
            'San Jose, California': 'San Jose, CA',
            'Los Angeles, California': 'Los Angeles, CA',
            'Highland, California': 'Highland, CA',
            'Torrance, California': 'Torrance, CA',
            'Long Beach, California': 'Long Beach, CA',
            'Gaithersburg, Maryland': 'Gaithersburg, MD',
            'Baltimore, Maryland': 'Baltimore, MD',
            'Milpitas, California': 'Milpitas, CA',
            'Bridgewater, Massachusetts': 'Bridgewater, MA',
            'Lafayette, Indiana': 'Lafayette, IN',
            'Kokomo, Indiana Area': 'Kokomo, IN',
            'Las Vegas, Nevada Area': 'Las Vegas, NV',
            'Cape Girardeau, Missouri': 'Cape Girardeau, MO',
            'Katy, Texas': 'Katy, TX',
            'Virginia Beach, Virginia': 'Virginia Beach, VA',
            'Monroe, Louisiana Area': 'Monroe, LA',
            'Jackson, Mississippi Area': 'Jackson, MS',
            'Greater Grand Rapids, Michigan Area': 'Grand Rapids, MI',
            'Baton Rouge, Louisiana Area': 'Baton Rouge, LA',
            'Myrtle Beach, South Carolina Area': 'Myrtle Beach, SC',
            'Chattanooga, Tennessee Area': 'Chattanooga, TN'
        }
    
    def _expand_abbreviations(self, text):
        if pd.isna(text) or text == '':
            return ''
        
        text = str(text)
        # Sort by length to avoid partial replacements
        for abbr, full in sorted(self.abbreviations.items(), key=lambda x: -len(x[0])):
            pattern = r'\b{}\b'.format(re.escape(abbr))
            text = re.sub(pattern, full, text, flags=re.IGNORECASE)
        
        return text

    def clean_location(self, location):

        if pd.isna(location) or location == '':
            return 'Unknown'
        
        location = str(location).strip()
        
        # Fix encoding issues
        try:
            # Try to fix common encoding problems
            if 'Ã¼' in location or 'Ã¶' in location or 'Ä°' in location:
                location = location.encode('latin1').decode('utf-8')
        except:
            pass
        
        # Apply direct mappings
        if location in self.location_mappings:
            location = self.location_mappings[location]
        
        # Handle Turkish patterns 
        if 'Ä°zmir' in location:
            location = 'Izmir, Turkey'
        if 'TÃ¼rkiye' in location:
            location = location.replace('TÃ¼rkiye', 'Turkey')
        if 'Ä°zmir, TÃ¼rkiye' in location:
            location = location.replace('Ä°zmir, TÃ¼rkiye', 'Izmir, Turkey') 
        if 'Amerika BirleÅŸik Devletleri' in location:
            location = 'United States'
        
        # Standardize "Area" suffix removal
        location = re.sub(r'\s+Area$', '', location)
        
        return location
    
    def preprocess(self, text, use_lemmatization=True):
        if pd.isna(text) or text == '':
            return ''
        
        text = str(text).lower()
        
        text = self._expand_abbreviations(text)
        
        text = re.sub(r'[^a-zA-Z\s]', ' ', text)
        
        text = ' '.join(text.split())
        
        try:
            words = word_tokenize(text)
        except LookupError:
            # Fallback to simple split if tokenizer fails
            words = text.split()

        processed_words = []
        for word in words:
            if word not in self.stop_words and len(word) > 1:
                if use_lemmatization and self.has_wordnet and self.lemmatizer:
                    # Use lemmatization if available
                    try:
                        lemmatized = self.lemmatizer.lemmatize(word)
                        processed_words.append(lemmatized)
                    except:
                        # Fall back to stemming
                        processed_words.append(self.stemmer.stem(word))
                else:
                    # Use stemming
                    processed_words.append(self.stemmer.stem(word))
        
        return ' '.join(processed_words)