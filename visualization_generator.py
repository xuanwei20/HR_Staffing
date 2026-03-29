import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import os

class VisualizationGenerator:
    
    def __init__(self, output_dir='results/figures'):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
    def generate_word_cloud(self, str_values, shape=(12, 9), background_col='gray', 
                           title=None, save_fig=True, filename='wordcloud.png', 
                           max_words=200):
        # Fetch words from the text
        text_wrd_cld = " ".join(str(item) for item in str_values if pd.notna(item))
        
        if not text_wrd_cld.strip():
            print("Warning: No text data to generate word cloud")
            return
        
        # Generate the word cloud
        word_cloud_params = {
            'collocations': False,
            'background_color': background_col,
            'colormap': matplotlib.cm.viridis,
            'width': 800,
            'height': 600,
            'max_words': max_words,
            'max_font_size': 100,
            'random_state': 42
        }
        
        word_cloud = WordCloud(**word_cloud_params).generate(text_wrd_cld)
        
        plt.figure(figsize=shape)
        plt.imshow(word_cloud, interpolation='bilinear')
        plt.axis("off")
        
        if title:
            plt.title(title, fontsize=16, pad=20)
        
        plt.tight_layout()
        
        if save_fig:
            save_path = os.path.join(self.output_dir, filename)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Word cloud saved to: {save_path}")
        
        # plt.show()
    
    def plot_top_locations(self, df, location_col='location', count_col='job_title', 
                          top_n=10, figsize=(10, 4), save_fig=True, filename='top_locations.png'):
        '''
        count_col : str
            Name of the column to use for counting
        '''
        if location_col not in df.columns:
            print(f"Error: Column '{location_col}' not found in DataFrame")
            return None
        
        # Create pivot table for location summary
        summary_by_loc = df.pivot_table(
            aggfunc='count', 
            index=location_col, 
            values=count_col,
            fill_value=0
        ).sort_values(by=count_col, ascending=False)
        
        # Add location column for plotting
        summary_by_loc['Locations'] = summary_by_loc.index.values
        
        # Print unique locations count
        print(f"\nUnique job locations: {len(summary_by_loc.index)}")
        
        # Create bar chart
        plt.figure(figsize=figsize)
        
        # Get top N locations
        top_locations = summary_by_loc.head(top_n)
        
        # Create bar chart
        bars = plt.bar(
            data=top_locations, 
            x='Locations', 
            height=count_col,
            color='skyblue',
            edgecolor='navy',
            alpha=0.7
        )
        
        # Customize the plot
        plt.xticks(rotation=45, ha='right')
        plt.title(f"Top {top_n} Locations of Candidates", fontsize=14, fontweight='bold')
        plt.xlabel("Locations", fontsize=12)
        plt.ylabel("Frequency", fontsize=12)
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(height)}',
                    ha='center', va='bottom')
        
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        
        if save_fig:
            save_path = os.path.join(self.output_dir, filename)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Top locations chart saved to: {save_path}")
        
        # plt.show()
        
        return summary_by_loc
    
    
    def plot_all_locations_wordcloud(self, df, location_col='location', 
                                     save_fig=True, filename='locations_wordcloud.png'):
        
        # Get all locations
        all_locations = df[location_col].dropna().values
        
        print(f"\nGenerating word cloud for ALL {len(all_locations)} locations...")
        
        # Generate word cloud with all locations
        self.generate_word_cloud(
            all_locations, 
            shape=(14, 10), 
            background_col="#b3cccc",
            title="Word Cloud of All Candidate Locations",
            save_fig=save_fig,
            filename=filename,
            max_words=300
        )
    
    def plot_job_titles_analysis(self, df, title_col='job_title', 
                                save_fig=True, filename_prefix='job_titles'):
        
        if title_col not in df.columns:
            print(f"Error: Column '{title_col}' not found in DataFrame")
            return
        
        # Clean job titles (remove NaN values)
        cleaned_titles = df[title_col].dropna().values
        
        print(f"\nTotal job titles: {len(cleaned_titles)}")
        print(f"Unique job titles: {df[title_col].nunique()}")
        
        # Get top job titles
        title_counts = df[title_col].value_counts().head(15)
        
        # Create bar chart for top job titles
        plt.figure(figsize=(12, 6))
        
        bars = plt.bar(
            range(len(title_counts)), 
            title_counts.values,
            color='lightcoral',
            edgecolor='darkred',
            alpha=0.7
        )
        
        plt.xticks(range(len(title_counts)), title_counts.index, rotation=45, ha='right')
        plt.title("Top 15 Job Titles", fontsize=14, fontweight='bold')
        plt.xlabel("Job Titles", fontsize=12)
        plt.ylabel("Frequency", fontsize=12)
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(height)}',
                    ha='center', va='bottom')
        
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        
        if save_fig:
            save_path = os.path.join(self.output_dir, f'top_{filename_prefix}.png')
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Top job titles chart saved to: {save_path}")
        
        # plt.show()
        
        # Generate word cloud for all job titles
        self.generate_word_cloud(
            cleaned_titles, 
            shape=(15, 10), 
            background_col="#b3cccc",
            title="Word Cloud of All Job Titles",
            save_fig=save_fig,
            filename=f'{filename_prefix}_wordcloud.png',
            max_words=300
        )
        
        return title_counts