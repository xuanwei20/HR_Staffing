from ranking_system import EnhancedCandidateRankingSystem
import os


def main():
    DATA_PATH = "data/potential-talents - Aspiring human resources - seeking human resources.csv"
    KEYWORDS = "aspiring human resources"
    
    print("="*60)
    print("ENHANCED CANDIDATE RANKING SYSTEM WITH RANKNET")
    print("="*60)
    print()
    
    system = EnhancedCandidateRankingSystem(KEYWORDS)
    
    system.load_and_preprocess_data(DATA_PATH)
    
    system.initial_ranking()
    
    print("\n" + "="*60)
    print("GENERATING VISUALIZATIONS")
    print("="*60)
    
    system.viz_generator.plot_top_locations(
        system.candidates_df, 
        location_col='location_cleaned',
        count_col='job_title',
        top_n=10
    )
    
    system.viz_generator.plot_all_locations_wordcloud(
        system.candidates_df,
        location_col='location_cleaned',
        save_fig=True,
        filename='locations_wordcloud.png'
    )
    
    system.viz_generator.plot_job_titles_analysis(
        system.candidates_df, 
        title_col='job_title'
    )
    
    # Interactive mode for starring candidates
    while True:
        print("\n" + "="*50)
        print("MAIN MENU")
        print("="*50)
        print("1. Star a candidate (provide feedback)")
        print("2. View candidate details")
        print("3. View current rankings")
        print("4. View starred candidates")
        print("5. Export results")
        print("6. Reset starred candidates")
        print("7. Exit")
        
        choice = input("\nEnter your choice (1-7): ").strip()
        
        if choice == '1':
            print("\n" + "-"*40)
            print("STAR A CANDIDATE")
            print("-"*40)
            
            print("\nCurrent Top 10 Candidates:")
            top_candidates = system.rankings[['id', 'job_title', 'fitness_score']].head(10)
            for _, row in top_candidates.iterrows():
                starred_marker = "★" if row['id'] in system.starred_candidates else " "
                print(f"  [{starred_marker}] ID: {row['id']:<8} | Score: {row['fitness_score']:.4f} | {row['job_title'][:50]}")
            
            cand_id = input("\nEnter candidate ID to star: ").strip()
            if cand_id.isdigit():
                cand_id_int = int(cand_id)
                if cand_id_int in system.candidates_df['id'].values:
                    system.star_candidate(cand_id_int)
                    
                    print("\n" + "-"*40)
                    print("UPDATED RANKINGS AFTER STARRING:")
                    print("-"*40)
                    updated_top = system.rankings[['id', 'job_title', 'fitness_score', 'rank']].head(10)
                    for _, row in updated_top.iterrows():
                        starred_marker = "★" if row['id'] in system.starred_candidates else " "
                        print(f"  [{starred_marker}] Rank {row['rank']:2d} | ID: {row['id']:<8} | Score: {row['fitness_score']:.4f}")
                else:
                    print(f"Error: Candidate ID {cand_id_int} not found")
            else:
                print("Please enter a valid numeric ID")
        
        elif choice == '2':
            print("\n" + "-"*40)
            print("VIEW CANDIDATE DETAILS")
            print("-"*40)
            cand_id = input("Enter candidate ID to view: ").strip()
            if cand_id.isdigit():
                cand_id_int = int(cand_id)
                if cand_id_int in system.candidates_df['id'].values:
                    system.get_candidate_details(cand_id_int)
                    
                    if cand_id_int in system.starred_candidates:
                        print(f"\n  ★ This candidate is STARRED")
                else:
                    print(f"Error: Candidate ID {cand_id_int} not found")
            else:
                print("Please enter a valid numeric ID")
        
        elif choice == '3':
            print("\n" + "-"*40)
            print("CURRENT RANKINGS")
            print("-"*40)
            
            n = input("How many candidates to display? (default: 20): ").strip()
            n = int(n) if n.isdigit() else 20
            
            print(f"\nTop {n} Candidates:")
            print("-"*80)
            
            display_cols = ['id', 'job_title', 'role_level', 'location_cleaned', 'fitness_score', 'rank']
            display_cols = [c for c in display_cols if c in system.rankings.columns]
            
            display_df = system.rankings[display_cols].head(n).copy()
            display_df['fitness_score'] = display_df['fitness_score'].round(4)
            display_df['starred'] = display_df['id'].apply(
                lambda x: '★' if x in system.starred_candidates else ''
            )
            
            print(display_df.to_string())
            
            print("\n" + "-"*40)
            print("SUMMARY STATISTICS:")
            print(f"Total candidates: {len(system.rankings)}")
            print(f"Starred candidates: {len(system.starred_candidates)}")
            print(f"Average fitness score: {system.rankings['fitness_score'].mean():.4f}")
            print(f"Median fitness score: {system.rankings['fitness_score'].median():.4f}")
        
        elif choice == '4':
            print("\n" + "-"*40)
            print("STARRED CANDIDATES")
            print("-"*40)
            
            if not system.starred_candidates:
                print("No candidates have been starred yet.")
            else:
                starred_df = system.rankings[system.rankings['id'].isin(system.starred_candidates)]
                starred_df = starred_df[['id', 'job_title', 'role_level', 'fitness_score', 'rank']].copy()
                starred_df['fitness_score'] = starred_df['fitness_score'].round(4)
                starred_df = starred_df.sort_values('rank')
                
                print(f"\nYou have starred {len(system.starred_candidates)} candidates:")
                print(starred_df.to_string())
        
        elif choice == '5':
            print("\n" + "-"*40)
            print("EXPORT RESULTS")
            print("-"*40)
            
            filename = input("Enter filename (default: enhanced_rankings.csv): ").strip()
            if not filename:
                filename = 'results/summary/enhanced_rankings.csv'
            
            print("\nExport options:")
            print("1. Full rankings (all columns)")
            print("2. Summary rankings (main columns only)")
            print("3. Starred candidates only")
            
            export_choice = input("Choose export format (1-3, default: 2): ").strip()
            
            if export_choice == '1':
                system.rankings.to_csv(filename, index=False)
                print(f"Full rankings exported to {filename}")
            elif export_choice == '3':
                starred_df = system.rankings[system.rankings['id'].isin(system.starred_candidates)]
                starred_df.to_csv(filename.replace('.csv', '_starred.csv'), index=False)
                print(f"Starred candidates exported to {filename.replace('.csv', '_starred.csv')}")
            else:
                system.export_results(filename)
        
        elif choice == '6':
            print("\n" + "-"*40)
            print("RESET STARRED CANDIDATES")
            print("-"*40)
            
            confirm = input("Are you sure you want to clear all starred candidates? (y/n): ").strip().lower()
            if confirm == 'y':
                system.starred_candidates = []
                system.ranknet_trained = False
                system.ranknet_model = None
                system.initial_ranking()
                print("All starred candidates have been cleared. Rankings reset to initial state.")
            else:
                print("Reset cancelled.")
        
        elif choice == '7':
            print("\n" + "="*50)
            print("EXITING SYSTEM")
            print("="*50)
            
            if system.starred_candidates:
                save = input("\nWould you like to export results before exiting? (y/n): ").strip().lower()
                if save == 'y':
                    filename = input("Enter filename (default: final_rankings.csv): ").strip()
                    if not filename:
                        filename = 'results/summary/final_rankings.csv'
                    system.export_results(filename)
            break
        
        else:
            print("Invalid choice, please try again")
    
    print("\nDone!")


if __name__ == "__main__":
    os.makedirs('results/summary', exist_ok=True)
    os.makedirs('results/figures', exist_ok=True)
    main()