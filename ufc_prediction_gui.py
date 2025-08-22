#!/usr/bin/env python3
"""
UFC Fight Prediction GUI Application
Allows users to select two fighters and get predictions using the trained ELO model
"""

import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext
import pandas as pd
import numpy as np
import joblib
import warnings
from typing import Dict, List, Optional, Tuple
import os

warnings.filterwarnings('ignore')

class UFCPredictionGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("UFC Fight Predictor - ELO Enhanced Model")
        self.root.geometry("1200x800")
        self.root.configure(bg='#2c3e50')
        
        # Load data and model
        self.fighter_mapping = None
        self.model = None
        self.training_data = None
        self.expected_features = None
        
        # Initialize the application
        self.load_data()
        self.create_widgets()
        
    def load_data(self):
        """Load all necessary data and model"""
        try:
            # Load fighter mapping
            self.fighter_mapping = pd.read_csv("dataUFC/fighter_id_mapping.csv")
            print(f"✓ Loaded {len(self.fighter_mapping)} fighters")
            
            # Load trained model
            self.model = joblib.load("models/random_forest_optimized_phase4_enhanced.joblib")
            print("✓ Loaded Random Forest Optimized model (99.59% accuracy)")
            
            # Load training data for feature reference
            self.training_data = pd.read_csv("dataUFC/ufc_engineered.csv")
            print(f"✓ Loaded training data: {self.training_data.shape}")
            
            # Get expected features from model
            if hasattr(self.model, 'feature_names_in_') and self.model.feature_names_in_ is not None:
                self.expected_features = list(self.model.feature_names_in_)
                print(f"✓ Model expects {len(self.expected_features)} features")
            else:
                raise ValueError("Model doesn't have feature names")
                
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load data: {str(e)}")
            self.root.destroy()
    
    def create_widgets(self):
        """Create the GUI widgets"""
        # Main frame
        main_frame = tk.Frame(self.root, bg='#2c3e50')
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Title
        title_label = tk.Label(main_frame, text="UFC Fight Predictor", 
                              font=("Arial", 24, "bold"), 
                              fg='#ecf0f1', bg='#2c3e50')
        title_label.pack(pady=(0, 20))
        
        subtitle_label = tk.Label(main_frame, text="ELO Enhanced Model (99.59% Accuracy)", 
                                 font=("Arial", 12), 
                                 fg='#bdc3c7', bg='#2c3e50')
        subtitle_label.pack(pady=(0, 20))
        
        # Create notebook for tabs
        notebook = ttk.Notebook(main_frame)
        notebook.pack(fill=tk.BOTH, expand=True)
        
        # Fighter Selection Tab
        self.create_fighter_selection_tab(notebook)
        
        # Fighter List Tab
        self.create_fighter_list_tab(notebook)
        
        # About Tab
        self.create_about_tab(notebook)
    
    def create_fighter_selection_tab(self, notebook):
        """Create the fighter selection and prediction tab"""
        selection_frame = ttk.Frame(notebook)
        notebook.add(selection_frame, text="Fight Prediction")
        
        # Fighter selection section
        selection_label = tk.Label(selection_frame, text="Select Two Fighters", 
                                  font=("Arial", 16, "bold"))
        selection_label.pack(pady=20)
        
        # Fighter 1 selection
        fighter1_frame = tk.Frame(selection_frame)
        fighter1_frame.pack(pady=10, padx=20, fill=tk.X)
        
        tk.Label(fighter1_frame, text="Fighter 1 (Red Corner):", 
                font=("Arial", 12, "bold")).pack(anchor=tk.W)
        
        self.fighter1_var = tk.StringVar()
        self.fighter1_entry = tk.Entry(fighter1_frame, textvariable=self.fighter1_var, 
                                      font=("Arial", 12), width=30)
        self.fighter1_entry.pack(side=tk.LEFT, padx=(0, 10))
        
        tk.Button(fighter1_frame, text="Search", 
                 command=lambda: self.search_fighter(1),
                 bg='#e74c3c', fg='white', font=("Arial", 10, "bold")).pack(side=tk.LEFT)
        
        # Fighter 2 selection
        fighter2_frame = tk.Frame(selection_frame)
        fighter2_frame.pack(pady=10, padx=20, fill=tk.X)
        
        tk.Label(fighter2_frame, text="Fighter 2 (Blue Corner):", 
                font=("Arial", 12, "bold")).pack(anchor=tk.W)
        
        self.fighter2_var = tk.StringVar()
        self.fighter2_entry = tk.Entry(fighter2_frame, textvariable=self.fighter2_var, 
                                      font=("Arial", 12), width=30)
        self.fighter2_entry.pack(side=tk.LEFT, padx=(0, 10))
        
        tk.Button(fighter2_frame, text="Search", 
                 command=lambda: self.search_fighter(2),
                 bg='#3498db', fg='white', font=("Arial", 10, "bold")).pack(side=tk.LEFT)
        
        # Prediction button
        predict_button = tk.Button(selection_frame, text="Predict Fight Outcome", 
                                  command=self.predict_fight,
                                  bg='#27ae60', fg='white', 
                                  font=("Arial", 14, "bold"),
                                  height=2, width=25)
        predict_button.pack(pady=30)
        
        # Results section
        results_frame = tk.Frame(selection_frame)
        results_frame.pack(pady=20, padx=20, fill=tk.BOTH, expand=True)
        
        tk.Label(results_frame, text="Prediction Results", 
                font=("Arial", 16, "bold")).pack()
        
        # Results text area
        self.results_text = scrolledtext.ScrolledText(results_frame, 
                                                     height=15, width=80,
                                                     font=("Courier", 10),
                                                     bg='#34495e', fg='#ecf0f1')
        self.results_text.pack(pady=10, fill=tk.BOTH, expand=True)
        
        # Clear results button
        clear_button = tk.Button(results_frame, text="Clear Results", 
                                command=self.clear_results,
                                bg='#95a5a6', fg='white',
                                font=("Arial", 10))
        clear_button.pack(pady=10)
    
    def create_fighter_list_tab(self, notebook):
        """Create the fighter list tab with search functionality"""
        list_frame = ttk.Frame(notebook)
        notebook.add(list_frame, text="Fighter Database")
        
        # Search section
        search_frame = tk.Frame(list_frame)
        search_frame.pack(pady=20, padx=20, fill=tk.X)
        
        tk.Label(search_frame, text="Search Fighters:", 
                font=("Arial", 14, "bold")).pack()
        
        self.search_var = tk.StringVar()
        self.search_var.trace('w', self.filter_fighters)
        search_entry = tk.Entry(search_frame, textvariable=self.search_var, 
                               font=("Arial", 12), width=40)
        search_entry.pack(pady=10)
        
        # Fighter list
        list_label_frame = tk.Frame(list_frame)
        list_label_frame.pack(pady=10, padx=20, fill=tk.X)
        
        tk.Label(list_label_frame, text="Fighter ID", width=10, 
                font=("Arial", 10, "bold")).pack(side=tk.LEFT)
        tk.Label(list_label_frame, text="Fighter Name", width=40, 
                font=("Arial", 10, "bold")).pack(side=tk.LEFT)
        
        # Create treeview for fighter list
        columns = ('ID', 'Name')
        self.fighter_tree = ttk.Treeview(list_frame, columns=columns, show='headings', height=20)
        
        self.fighter_tree.heading('ID', text='Fighter ID')
        self.fighter_tree.heading('Name', text='Fighter Name')
        
        self.fighter_tree.column('ID', width=100, anchor=tk.CENTER)
        self.fighter_tree.column('Name', width=400, anchor=tk.W)
        
        # Scrollbar for treeview
        scrollbar = ttk.Scrollbar(list_frame, orient=tk.VERTICAL, command=self.fighter_tree.yview)
        self.fighter_tree.configure(yscrollcommand=scrollbar.set)
        
        self.fighter_tree.pack(pady=10, padx=20, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Populate fighter list
        self.populate_fighter_list()
        
        # Double-click to select fighter
        self.fighter_tree.bind('<Double-1>', self.select_fighter_from_list)
    
    def create_about_tab(self, notebook):
        """Create the about tab with model information"""
        about_frame = ttk.Frame(notebook)
        notebook.add(about_frame, text="About")
        
        about_text = """
UFC Fight Predictor - ELO Enhanced Model

Model Information:
• Algorithm: Random Forest Classifier (Optimized)
• Accuracy: 99.59%
• Features: 121 total (including 9 ELO features)
• Training Data: UFC fights with comprehensive statistics

ELO System Features:
• Base ELO ratings for overall skill
• Weight class-specific ELO ratings
• ELO gradient tracking (trend analysis)
• Head-to-head records
• Rolling statistics for recent fights

How to Use:
1. Go to "Fighter Database" tab to browse fighters
2. Search for specific fighters by name
3. Double-click a fighter to select them
4. Switch to "Fight Prediction" tab
5. Enter fighter IDs or use search function
6. Click "Predict Fight Outcome" to get results

The model considers:
• Fighter statistics and records
• ELO ratings and trends
• Physical attributes
• Fight context and history
• Recent performance metrics

Note: Predictions are based on historical data and should be used for entertainment purposes only.
        """
        
        about_label = tk.Label(about_frame, text=about_text, 
                              font=("Arial", 11), 
                              justify=tk.LEFT, 
                              wraplength=800)
        about_label.pack(pady=50, padx=50)
    
    def populate_fighter_list(self):
        """Populate the fighter treeview"""
        # Clear existing items
        for item in self.fighter_tree.get_children():
            self.fighter_tree.delete(item)
        
        # Add all fighters
        for _, row in self.fighter_mapping.iterrows():
            self.fighter_tree.insert('', tk.END, 
                                   values=(row['FighterID'], row['FighterName']))
    
    def filter_fighters(self, *args):
        """Filter fighters based on search term"""
        search_term = self.search_var.get().lower()
        
        # Clear existing items
        for item in self.fighter_tree.get_children():
            self.fighter_tree.delete(item)
        
        # Add filtered fighters
        for _, row in self.fighter_mapping.iterrows():
            if search_term in row['FighterName'].lower():
                self.fighter_tree.insert('', tk.END, 
                                       values=(row['FighterID'], row['FighterName']))
    
    def select_fighter_from_list(self, event):
        """Select a fighter from the list"""
        selection = self.fighter_tree.selection()
        if selection:
            item = self.fighter_tree.item(selection[0])
            fighter_id = item['values'][0]
            fighter_name = item['values'][1]
            
            # Ask user which fighter slot to fill
            dialog = tk.Toplevel(self.root)
            dialog.title("Select Fighter Slot")
            dialog.geometry("300x150")
            dialog.transient(self.root)
            dialog.grab_set()
            
            tk.Label(dialog, text=f"Select slot for {fighter_name} (ID: {fighter_id})", 
                    font=("Arial", 12)).pack(pady=20)
            
            def set_fighter1():
                self.fighter1_var.set(f"{fighter_name} (ID: {fighter_id})")
                dialog.destroy()
            
            def set_fighter2():
                self.fighter2_var.set(f"{fighter_name} (ID: {fighter_id})")
                dialog.destroy()
            
            tk.Button(dialog, text="Fighter 1 (Red)", command=set_fighter1, 
                     bg='#e74c3c', fg='white').pack(side=tk.LEFT, padx=10, pady=20)
            tk.Button(dialog, text="Fighter 2 (Blue)", command=set_fighter2, 
                     bg='#3498db', fg='white').pack(side=tk.RIGHT, padx=10, pady=20)
    
    def search_fighter(self, fighter_num):
        """Search for a fighter by name"""
        search_term = self.fighter1_var.get() if fighter_num == 1 else self.fighter2_var.get()
        
        if not search_term:
            messagebox.showwarning("Warning", "Please enter a fighter name to search")
            return
        
        # Search in fighter mapping
        matches = self.fighter_mapping[
            self.fighter_mapping['FighterName'].str.contains(search_term, case=False, na=False)
        ]
        
        if len(matches) == 0:
            messagebox.showinfo("No Results", f"No fighters found matching '{search_term}'")
            return
        
        if len(matches) == 1:
            # Single match, auto-fill
            fighter = matches.iloc[0]
            fighter_text = f"{fighter['FighterName']} (ID: {fighter['FighterID']})"
            if fighter_num == 1:
                self.fighter1_var.set(fighter_text)
            else:
                self.fighter2_var.set(fighter_text)
        else:
            # Multiple matches, show selection dialog
            self.show_fighter_selection_dialog(matches, fighter_num)
    
    def show_fighter_selection_dialog(self, matches, fighter_num):
        """Show dialog to select from multiple fighter matches"""
        dialog = tk.Toplevel(self.root)
        dialog.title("Select Fighter")
        dialog.geometry("400x300")
        dialog.transient(self.root)
        dialog.grab_set()
        
        tk.Label(dialog, text="Multiple fighters found. Please select one:", 
                font=("Arial", 12, "bold")).pack(pady=10)
        
        # Create listbox for selection
        listbox = tk.Listbox(dialog, font=("Arial", 11), height=10)
        listbox.pack(pady=10, padx=20, fill=tk.BOTH, expand=True)
        
        # Populate listbox
        for _, fighter in matches.iterrows():
            listbox.insert(tk.END, f"{fighter['FighterName']} (ID: {fighter['FighterID']})")
        
        def select_fighter():
            selection = listbox.curselection()
            if selection:
                fighter_text = listbox.get(selection[0])
                if fighter_num == 1:
                    self.fighter1_var.set(fighter_text)
                else:
                    self.fighter2_var.set(fighter_text)
                dialog.destroy()
        
        tk.Button(dialog, text="Select", command=select_fighter, 
                 bg='#27ae60', fg='white').pack(pady=10)
    
    def extract_fighter_id(self, fighter_text):
        """Extract fighter ID from fighter text"""
        try:
            # Extract ID from format "Name (ID: X)"
            if "(ID:" in fighter_text:
                id_part = fighter_text.split("(ID:")[1].split(")")[0].strip()
                return int(id_part)
            else:
                # Try to parse as direct ID
                return int(fighter_text)
        except:
            return None
    
    def predict_fight(self):
        """Predict the outcome of a fight between two fighters"""
        # Get fighter IDs
        fighter1_text = self.fighter1_var.get()
        fighter2_text = self.fighter2_var.get()
        
        if not fighter1_text or not fighter2_text:
            messagebox.showwarning("Warning", "Please select both fighters")
            return
        
        fighter1_id = self.extract_fighter_id(fighter1_text)
        fighter2_id = self.extract_fighter_id(fighter2_text)
        
        if fighter1_id is None or fighter2_id is None:
            messagebox.showerror("Error", "Invalid fighter ID format")
            return
        
        if fighter1_id == fighter2_id:
            messagebox.showwarning("Warning", "Please select two different fighters")
            return
        
        # Get fighter names
        fighter1_name = self.fighter_mapping[self.fighter_mapping['FighterID'] == fighter1_id]['FighterName'].iloc[0]
        fighter2_name = self.fighter_mapping[self.fighter_mapping['FighterID'] == fighter2_id]['FighterName'].iloc[0]
        
        # Clear previous results
        self.results_text.delete(1.0, tk.END)
        
        # Show prediction in progress
        self.results_text.insert(tk.END, "Analyzing fight data...\n")
        self.results_text.insert(tk.END, f"Fighter 1 (Red): {fighter1_name} (ID: {fighter1_id})\n")
        self.results_text.insert(tk.END, f"Fighter 2 (Blue): {fighter2_name} (ID: {fighter2_id})\n")
        self.results_text.insert(tk.END, "=" * 60 + "\n\n")
        
        # Make prediction
        try:
            result = self.make_prediction(fighter1_id, fighter2_id, fighter1_name, fighter2_name)
            
            if result:
                self.display_prediction_result(result, fighter1_name, fighter2_name)
            else:
                self.results_text.insert(tk.END, "❌ Prediction failed - insufficient data for one or both fighters\n")
                
        except Exception as e:
            self.results_text.insert(tk.END, f"❌ Error during prediction: {str(e)}\n")
    
    def make_prediction(self, fighter1_id, fighter2_id, fighter1_name, fighter2_name):
        """Make prediction using the trained model"""
        # Find fighter data in training dataset
        fighter1_data = self.training_data[
            (self.training_data['RedFighter'] == fighter1_name) | 
            (self.training_data['BlueFighter'] == fighter1_name)
        ]
        
        fighter2_data = self.training_data[
            (self.training_data['RedFighter'] == fighter2_name) | 
            (self.training_data['BlueFighter'] == fighter2_name)
        ]
        
        if len(fighter1_data) == 0 or len(fighter2_data) == 0:
            return None
        
        # Use most recent data for each fighter
        fighter1_stats = fighter1_data.iloc[-1]
        fighter2_stats = fighter2_data.iloc[-1]
        
        # Prepare features
        features = self.prepare_fight_features(fighter1_stats, fighter2_stats, fighter1_name, fighter2_name)
        
        if features is None:
            return None
        
        # Make prediction
        prediction = self.model.predict([features])[0]
        probabilities = self.model.predict_proba([features])[0]
        
        # Determine winner
        if prediction == 1:  # Red fighter wins
            winner = fighter1_name
            winner_type = "Red"
            confidence = max(probabilities)
            red_prob = probabilities[1]
            blue_prob = probabilities[0]
        else:  # Blue fighter wins
            winner = fighter2_name
            winner_type = "Blue"
            confidence = max(probabilities)
            red_prob = probabilities[0]
            blue_prob = probabilities[1]
        
        return {
            'winner': winner,
            'winner_type': winner_type,
            'confidence': confidence,
            'red_probability': red_prob,
            'blue_probability': blue_prob,
            'prediction': prediction
        }
    
    def prepare_fight_features(self, fighter1_stats, fighter2_stats, fighter1_name, fighter2_name):
        """Prepare features for prediction"""
        try:
            # Initialize feature dictionary with zeros
            features = {feature: 0.0 for feature in self.expected_features}
            
            # Determine which fighter is Red and which is Blue
            fighter1_is_red = fighter1_stats['RedFighter'] == fighter1_name
            fighter2_is_red = fighter2_stats['RedFighter'] == fighter2_name
            
            # Get Red and Blue fighter stats
            if fighter1_is_red:
                red_stats = fighter1_stats
                blue_stats = fighter2_stats
            elif fighter2_is_red:
                red_stats = fighter2_stats
                blue_stats = fighter1_stats
            else:
                # Neither fighter is Red in their stats, use fighter1 as Red
                red_stats = fighter1_stats
                blue_stats = fighter2_stats
            
            # Calculate differences for base features (non-ELO)
            base_features = [f for f in self.expected_features if not f.startswith('ELO')]
            
            for feature in base_features:
                if feature.endswith('Dif'):
                    # Handle difference features
                    base_feature = feature.replace('Dif', '')
                    red_feature = 'Red' + base_feature
                    blue_feature = 'Blue' + base_feature
                    
                    if red_feature in red_stats and blue_feature in blue_stats:
                        features[feature] = red_stats[red_feature] - blue_stats[blue_feature]
                
                elif feature.endswith('_diff'):
                    # Handle other difference features
                    base_feature = feature.replace('_diff', '')
                    red_feature = 'Red' + base_feature.replace('_', '')
                    blue_feature = 'Blue' + base_feature.replace('_', '')
                    
                    if red_feature in red_stats and blue_feature in blue_stats:
                        features[feature] = red_stats[red_feature] - blue_stats[blue_feature]
                
                elif feature.endswith('_ratio'):
                    # Handle ratio features
                    base_feature = feature.replace('_ratio', '')
                    red_feature = 'Red' + base_feature.replace('_', '')
                    blue_feature = 'Blue' + base_feature.replace('_', '')
                    
                    if red_feature in red_stats and blue_feature in blue_stats:
                        if blue_stats[blue_feature] != 0:
                            features[feature] = red_stats[red_feature] / blue_stats[blue_feature]
                        else:
                            features[feature] = 1.0 if red_stats[red_feature] == 0 else 999.0
                
                else:
                    # Handle other features (copy from Red fighter)
                    red_feature = 'Red' + feature
                    if red_feature in red_stats:
                        features[feature] = red_stats[red_feature]
            
            # Handle ELO features
            elo_features = [f for f in self.expected_features if f.startswith('ELO')]
            for feature in elo_features:
                if feature in red_stats:
                    features[feature] = red_stats[feature]
            
            # Convert to list in correct order
            feature_list = [features[feature] for feature in self.expected_features]
            
            return feature_list
            
        except Exception as e:
            print(f"Error preparing features: {e}")
            return None
    
    def display_prediction_result(self, result, fighter1_name, fighter2_name):
        """Display prediction results in the text area"""
        self.results_text.insert(tk.END, "🏆 PREDICTION RESULTS 🏆\n")
        self.results_text.insert(tk.END, "=" * 60 + "\n\n")
        
        self.results_text.insert(tk.END, f"FIGHT: {fighter1_name} vs {fighter2_name}\n\n")
        
        self.results_text.insert(tk.END, f"🏆 PREDICTED WINNER: {result['winner']}\n")
        self.results_text.insert(tk.END, f"CONFIDENCE: {result['confidence']:.1%}\n")
        self.results_text.insert(tk.END, f"WINNER TYPE: {result['winner_type']} Corner\n\n")
        
        self.results_text.insert(tk.END, "PROBABILITY BREAKDOWN:\n")
        self.results_text.insert(tk.END, f"   {fighter1_name} (Red): {result['red_probability']:.1%}\n")
        self.results_text.insert(tk.END, f"   {fighter2_name} (Blue): {result['blue_probability']:.1%}\n\n")
        
        # Add confidence interpretation
        confidence = result['confidence']
        if confidence >= 0.8:
            confidence_level = "Very High"
        elif confidence >= 0.6:
            confidence_level = "High"
        elif confidence >= 0.4:
            confidence_level = "Medium"
        else:
            confidence_level = "Low"
        
        self.results_text.insert(tk.END, f"CONFIDENCE LEVEL: {confidence_level}\n")
        
        if confidence >= 0.7:
            self.results_text.insert(tk.END, "✅ Strong prediction based on comprehensive data\n")
        elif confidence >= 0.5:
            self.results_text.insert(tk.END, "⚠️ Moderate prediction - consider other factors\n")
        else:
            self.results_text.insert(tk.END, "Low confidence - fight could go either way\n")
        
        self.results_text.insert(tk.END, "\n" + "=" * 60 + "\n")
        self.results_text.insert(tk.END, "Model: Random Forest Optimized (ELO Enhanced)\n")
        self.results_text.insert(tk.END, "Accuracy: 99.59%\n")
        self.results_text.insert(tk.END, "Features: 121 (including 9 ELO features)\n")
    
    def clear_results(self):
        """Clear the results text area"""
        self.results_text.delete(1.0, tk.END)

def main():
    """Main function to run the GUI application"""
    root = tk.Tk()
    app = UFCPredictionGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()

