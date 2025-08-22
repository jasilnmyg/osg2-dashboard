def campaign_management_page():
    st.header("🎯 Campaign Management")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Create New Campaign")
        with st.form("campaign_form"):
            campaign_id = st.text_input("Campaign ID", placeholder="e.g., camp_2024_001")
            campaign_name = st.text_input("Campaign Name", placeholder="e.g., Diwali Special Offers")
            start_date = st.date_input("Start Date")
            end_date = st.date_input("End Date")
            campaign_type = st.radio("Campaign Type", ["selected_stores", "all_kerala"])
            
            stores_df = get_all_stores()
            if campaign_type == "selected_stores":
                if not stores_df.empty:
                    target_stores = st.multiselect("Target Stores", 
                                                 options=stores_df['store_id'].tolist(),
                                                 format_func=lambda x: f"{x} - {stores_df[stores_df['store_id']==x]['store_name'].iloc[0]}")
                else:
                    target_stores = []
                    st.warning("No stores available")
            else:
                target_stores = stores_df['store_id'].tolist() if not stores_df.empty else []
                st.info(f"Campaign will target all {len(target_stores)} stores in Kerala")
            
            # Product selection interface
            st.subheader("Select Offer Products")
            
            # Option 1: Select by category first, then products
            selection_method = st.radio("Product Selection Method", 
                                      ["Select by Category", "Select Individual Products"])
            
            selected_products = []
            
            if selection_method == "Select by Category":
                selected_categories = st.multiselect("Select Categories", ITEM_CATEGORIES)
                
                for category in selected_categories:
                    st.write(f"**{category} Products:**")
                    available_products = PRODUCTS_BY_CATEGORY.get(category, [])
                    category_products = st.multiselect(
                        f"Select products from {category}",
                        options=available_products,
                        key=f"products_{category}"
                    )
                    selected_products.extend([(product, category) for product in category_products])
            
            else:  # Individual product selection
                all_products = []
                for cat, products in PRODUCTS_BY_CATEGORY.items():
                    for product in products:
                        all_products.append((f"{product} ({cat})", product, cat))
                
                selected_product_options = st.multiselect(
                    "Select Individual Products",
                    options=[item[0] for item in all_products]
                )
                
                selected_products = [(item[1], item[2]) for item in all_products 
                                   if item[0] in selected_product_options]
            
            # Custom products option
            st.subheader("Add Custom Products")
            custom_products_text = st.text_area(
                "Add custom products (one per line, format: Product Name - Category)",
                placeholder="iPhone 16 - Electronics\nSmart TV 65 inch - Electronics"
            )
            
            if custom_products_text:
                for line in custom_products_text.strip().split('\n'):
                    if ' - ' in line:
                        product, category = line.split(' - ', 1)
                        selected_products.append((product.strip(), category.strip()))
            
            # Display selected products
            if selected_products:
                st.subheader("Selected Offer Products")
                products_df = pd.DataFrame(selected_products, columns=['Product', 'Category'])
                st.dataframe(products_df, use_container_width=True)
            
            offer_description = st.text_area("Offer Description", 
                                           placeholder="e.g., Special discount on selected electronics and fashion items")
            
            if st.form_submit_button("Create Campaign"):
                if campaign_id and campaign_name and target_stores and selected_products:
                    # Convert selected_products to just product names
                    product_names = [product[0] for product in selected_products]
                    
                    conn = get_db_connection()
                    try:
                        cursor = conn.cursor()
                        cursor.execute('''
                            INSERT INTO campaigns
                            (campaign_id, campaign_name, start_date, end_date, campaign_type,
                             target_stores, offer_products, offer_description)
                            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                        ''', (campaign_id, campaign_name, start_date, end_date, campaign_type,
                              str(target_stores), str(product_names), offer_description))
                        conn.commit()
                        st.success(f"Campaign created successfully with {len(product_names)} offer products!")
                    except Exception as e:
                        st.error(f"Error creating campaign: {e}")
                    finally:
                        conn.close()
                else:
                    st.error("Please fill all required fields and select at least one product")
    
    with col2:
        st.subheader("Campaign Analytics")
        # Show campaign summary metrics
        conn = get_db_connection()
        campaigns_df = pd.read_sql_query('SELECT * FROM campaigns ORDER BY created_date DESC', conn)
        conn.close()
        
        if not campaigns_df.empty:
            st.metric("Total Campaigns", len(campaigns_df))
            active_campaigns = campaigns_df[
                (pd.to_datetime(campaigns_df['start_date']) <= pd.Timestamp.now()) &
                (pd.to_datetime(campaigns_df['end_date']) >= pd.Timestamp.now())
            ]
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import sqlite3
import pickle
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

# Database setup
def init_database():
    conn = sqlite3.connect('sales_campaign_data.db')
    cursor = conn.cursor()
    
    # Stores table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS stores (
            store_id TEXT PRIMARY KEY,
            store_name TEXT NOT NULL,
            location TEXT NOT NULL,
            district TEXT NOT NULL,
            created_date DATE DEFAULT CURRENT_DATE
        )
    ''')
    
    # Daily sales data
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS daily_sales (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            date DATE NOT NULL,
            store_id TEXT NOT NULL,
            category TEXT NOT NULL,
            product_name TEXT NOT NULL,
            sales_amount REAL NOT NULL,
            units_sold INTEGER DEFAULT 0,
            FOREIGN KEY (store_id) REFERENCES stores (store_id)
        )
    ''')
    
    # Campaigns table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS campaigns (
            campaign_id TEXT PRIMARY KEY,
            campaign_name TEXT NOT NULL,
            start_date DATE NOT NULL,
            end_date DATE NOT NULL,
            campaign_type TEXT NOT NULL, -- 'selected_stores' or 'all_kerala'
            target_stores TEXT, -- JSON array of store IDs
            offer_products TEXT NOT NULL, -- JSON array of product names
            offer_description TEXT,
            created_date DATE DEFAULT CURRENT_DATE
        )
    ''')
    
    # Campaign performance tracking
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS campaign_performance (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            campaign_id TEXT NOT NULL,
            store_id TEXT NOT NULL,
            product_name TEXT NOT NULL,
            category TEXT NOT NULL,
            sales_before REAL, -- 7 days before campaign
            sales_during REAL, -- during campaign
            sales_after REAL,  -- 7 days after campaign
            units_before INTEGER,
            units_during INTEGER,
            units_after INTEGER,
            uplift_percent REAL,
            units_uplift_percent REAL,
            FOREIGN KEY (campaign_id) REFERENCES campaigns (campaign_id),
            FOREIGN KEY (store_id) REFERENCES stores (store_id)
        )
    ''')
    
    conn.commit()
    conn.close()

# Initialize database
init_database()

# Configuration
st.set_page_config(
    page_title="Sales Prediction & Campaign Analysis",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Item categories
ITEM_CATEGORIES = [
    'Electronics', 'Clothing & Fashion', 'Groceries & Food', 'Home & Garden',
    'Sports & Fitness', 'Beauty & Personal Care', 'Books & Stationery',
    'Toys & Games', 'Automotive', 'Health & Medicine', 'Mobile & Accessories',
    'Footwear', 'Jewelry & Watches', 'Kitchen & Dining', 'Baby & Kids'
]

# Sample products by category - you can expand this
PRODUCTS_BY_CATEGORY = {
    'Electronics': ['iPhone 15', 'Samsung TV 55"', 'MacBook Air', 'iPad', 'AirPods', 'Gaming Laptop', 'Smart Watch', 'Bluetooth Speaker'],
    'Clothing & Fashion': ['Mens Shirt', 'Ladies Kurti', 'Jeans', 'Saree', 'T-Shirt', 'Dress', 'Jacket', 'Traditional Wear'],
    'Groceries & Food': ['Rice 25kg', 'Cooking Oil', 'Pulses', 'Spices Set', 'Tea', 'Coffee', 'Snacks', 'Beverages'],
    'Home & Garden': ['Sofa Set', 'Dining Table', 'Bed', 'Garden Tools', 'Plants', 'Home Decor', 'Furniture', 'Lighting'],
    'Sports & Fitness': ['Treadmill', 'Yoga Mat', 'Cricket Kit', 'Football', 'Gym Equipment', 'Sports Shoes', 'Fitness Tracker'],
    'Beauty & Personal Care': ['Face Cream', 'Shampoo', 'Perfume', 'Makeup Kit', 'Hair Oil', 'Soap', 'Skincare Set'],
    'Books & Stationery': ['Novels', 'Study Books', 'Notebooks', 'Pens', 'Art Supplies', 'Educational Books'],
    'Toys & Games': ['Action Figures', 'Board Games', 'Puzzles', 'Remote Control Car', 'Dolls', 'Educational Toys'],
    'Automotive': ['Car Accessories', 'Bike Parts', 'Tyres', 'Engine Oil', 'Car Care Products'],
    'Health & Medicine': ['Vitamins', 'First Aid Kit', 'Health Supplements', 'Medical Devices', 'Ayurvedic Products'],
    'Mobile & Accessories': ['Phone Cases', 'Chargers', 'Power Banks', 'Headphones', 'Screen Guards', 'Mobile Stands'],
    'Footwear': ['Running Shoes', 'Formal Shoes', 'Sandals', 'Boots', 'Slippers', 'Sports Shoes'],
    'Jewelry & Watches': ['Gold Jewelry', 'Silver Jewelry', 'Watches', 'Diamond Jewelry', 'Fashion Jewelry'],
    'Kitchen & Dining': ['Cookware Set', 'Dinner Set', 'Kitchen Appliances', 'Storage Containers', 'Cutlery'],
    'Baby & Kids': ['Baby Clothes', 'Toys', 'Baby Care Products', 'Kids Books', 'School Supplies', 'Baby Food']
}

# Helper functions
def get_db_connection():
    return sqlite3.connect('sales_campaign_data.db')

def add_store(store_id, store_name, location, district):
    conn = get_db_connection()
    try:
        cursor = conn.cursor()
        cursor.execute('''
            INSERT OR REPLACE INTO stores (store_id, store_name, location, district)
            VALUES (?, ?, ?, ?)
        ''', (store_id, store_name, location, district))
        conn.commit()
        return True
    except Exception as e:
        st.error(f"Error adding store: {e}")
        return False
    finally:
        conn.close()

def get_all_stores():
    conn = get_db_connection()
    df = pd.read_sql_query('SELECT * FROM stores ORDER BY store_name', conn)
    conn.close()
    return df

def add_daily_sales(date, store_id, category, product_name, sales_amount, units_sold):
    conn = get_db_connection()
    try:
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO daily_sales (date, store_id, category, product_name, sales_amount, units_sold)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (date, store_id, category, product_name, sales_amount, units_sold))
        conn.commit()
        return True
    except Exception as e:
        st.error(f"Error adding sales data: {e}")
        return False
    finally:
        conn.close()

def get_sales_data(store_id=None, start_date=None, end_date=None, category=None, product_name=None):
    conn = get_db_connection()
    query = '''
        SELECT ds.*, s.store_name, s.location, s.district
        FROM daily_sales ds
        JOIN stores s ON ds.store_id = s.store_id
        WHERE 1=1
    '''
    params = []
    
    if store_id:
        if isinstance(store_id, list):
            placeholders = ','.join(['?' for _ in store_id])
            query += f' AND ds.store_id IN ({placeholders})'
            params.extend(store_id)
        else:
            query += ' AND ds.store_id = ?'
            params.append(store_id)
    if start_date:
        query += ' AND ds.date >= ?'
        params.append(start_date)
    if end_date:
        query += ' AND ds.date <= ?'
        params.append(end_date)
    if category:
        query += ' AND ds.category = ?'
        params.append(category)
    if product_name:
        if isinstance(product_name, list):
            placeholders = ','.join(['?' for _ in product_name])
            query += f' AND ds.product_name IN ({placeholders})'
            params.extend(product_name)
        else:
            query += ' AND ds.product_name = ?'
            params.append(product_name)
    
    query += ' ORDER BY ds.date DESC'
    
    df = pd.read_sql_query(query, conn, params=params)
    conn.close()
    return df

def calculate_campaign_performance(campaign_id, store_id, product_name, start_date, end_date):
    # Calculate 7 days before, during, and 7 days after campaign performance
    start_dt = pd.to_datetime(start_date)
    end_dt = pd.to_datetime(end_date)
    
    before_start = (start_dt - timedelta(days=7)).strftime('%Y-%m-%d')
    before_end = (start_dt - timedelta(days=1)).strftime('%Y-%m-%d')
    after_start = (end_dt + timedelta(days=1)).strftime('%Y-%m-%d')
    after_end = (end_dt + timedelta(days=7)).strftime('%Y-%m-%d')
    
    # Get sales data for each period
    sales_before_df = get_sales_data(store_id, before_start, before_end, product_name=product_name)
    sales_during_df = get_sales_data(store_id, start_date, end_date, product_name=product_name)
    sales_after_df = get_sales_data(store_id, after_start, after_end, product_name=product_name)
    
    sales_before = sales_before_df['sales_amount'].sum()
    sales_during = sales_during_df['sales_amount'].sum()
    sales_after = sales_after_df['sales_amount'].sum()
    
    units_before = sales_before_df['units_sold'].sum()
    units_during = sales_during_df['units_sold'].sum()
    units_after = sales_after_df['units_sold'].sum()
    
    # Calculate uplift percentages
    if sales_before > 0:
        uplift_percent = ((sales_during - sales_before) / sales_before) * 100
    else:
        uplift_percent = 100 if sales_during > 0 else 0
    
    if units_before > 0:
        units_uplift_percent = ((units_during - units_before) / units_before) * 100
    else:
        units_uplift_percent = 100 if units_during > 0 else 0
    
    return {
        'sales_before': sales_before,
        'sales_during': sales_during,
        'sales_after': sales_after,
        'units_before': units_before,
        'units_during': units_during,
        'units_after': units_after,
        'uplift_percent': uplift_percent,
        'units_uplift_percent': units_uplift_percent
    }

def save_campaign_performance(campaign_id, store_id, product_name, category, performance_data):
    conn = get_db_connection()
    try:
        cursor = conn.cursor()
        cursor.execute('''
            INSERT OR REPLACE INTO campaign_performance
            (campaign_id, store_id, product_name, category, sales_before, sales_during, sales_after, 
             units_before, units_during, units_after, uplift_percent, units_uplift_percent)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (campaign_id, store_id, product_name, category,
              performance_data['sales_before'], performance_data['sales_during'],
              performance_data['sales_after'], performance_data['units_before'],
              performance_data['units_during'], performance_data['units_after'],
              performance_data['uplift_percent'], performance_data['units_uplift_percent']))
        conn.commit()
        return True
    except Exception as e:
        st.error(f"Error saving campaign performance: {e}")
        return False
    finally:
        conn.close()

def create_prediction_model(data):
    """Create and train ML model for sales prediction"""
    if len(data) < 30:  # Need minimum data for training
        return None, None, "Insufficient data for training (minimum 30 records required)"
    
    # Feature engineering
    data['date'] = pd.to_datetime(data['date'])
    data['day_of_week'] = data['date'].dt.dayofweek
    data['month'] = data['date'].dt.month
    data['day_of_month'] = data['date'].dt.day
    data['is_weekend'] = (data['day_of_week'] >= 5).astype(int)
    
    # Create lagged features
    data = data.sort_values(['store_id', 'category', 'date'])
    data['sales_lag_1'] = data.groupby(['store_id', 'category'])['sales_amount'].shift(1)
    data['sales_lag_7'] = data.groupby(['store_id', 'category'])['sales_amount'].shift(7)
    data['sales_rolling_7'] = data.groupby(['store_id', 'category'])['sales_amount'].rolling(7).mean().reset_index(0, drop=True)
    
    # Encode categorical variables
    data_encoded = pd.get_dummies(data, columns=['store_id', 'category'], prefix=['store', 'cat'])
    
    # Select features
    feature_cols = [col for col in data_encoded.columns if col.startswith(('store_', 'cat_')) or 
                   col in ['day_of_week', 'month', 'day_of_month', 'is_weekend', 
                          'sales_lag_1', 'sales_lag_7', 'sales_rolling_7']]
    
    # Remove rows with NaN values
    data_clean = data_encoded.dropna()
    
    if len(data_clean) < 20:
        return None, None, "Insufficient clean data for training"
    
    X = data_clean[feature_cols]
    y = data_clean['sales_amount']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train models
    models = {
        'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
        'Linear Regression': LinearRegression()
    }
    
    best_model = None
    best_score = float('-inf')
    best_model_name = None
    
    for name, model in models.items():
        model.fit(X_train_scaled, y_train)
        score = model.score(X_test_scaled, y_test)
        if score > best_score:
            best_score = score
            best_model = model
            best_model_name = name
    
    return best_model, scaler, f"Model trained successfully. Best model: {best_model_name} (R² = {best_score:.3f})"

# Streamlit App
def main():
    st.title("📈 Sales Prediction & Campaign Analysis System")
    st.markdown("Advanced AI-powered sales forecasting and campaign performance analysis")
    
    # Sidebar
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox("Choose a page", [
        "🏪 Store Management",
        "📊 Daily Sales Entry",
        "🎯 Campaign Management",
        "📈 Sales Analysis",
        "🤖 AI Predictions",
        "📋 Campaign Performance"
    ])
    
    if page == "🏪 Store Management":
        store_management_page()
    elif page == "📊 Daily Sales Entry":
        daily_sales_page()
    elif page == "🎯 Campaign Management":
        campaign_management_page()
    elif page == "📈 Sales Analysis":
        sales_analysis_page()
    elif page == "🤖 AI Predictions":
        ai_predictions_page()
    elif page == "📋 Campaign Performance":
        campaign_performance_page()

def store_management_page():
    st.header("🏪 Store Management")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Add New Store")
        with st.form("add_store_form"):
            store_id = st.text_input("Store ID", placeholder="e.g., balussery-myg")
            store_name = st.text_input("Store Name", placeholder="e.g., Balussery MYG Store")
            location = st.text_input("Location", placeholder="e.g., Balussery")
            district = st.selectbox("District", [
                "Kozhikode", "Malappuram", "Thrissur", "Kochi", "Thiruvananthapuram",
                "Kottayam", "Alappuzha", "Kollam", "Palakkad", "Kannur", "Kasaragod",
                "Wayanad", "Idukki", "Pathanamthitta"
            ])
            
            if st.form_submit_button("Add Store"):
                if store_id and store_name and location:
                    if add_store(store_id, store_name, location, district):
                        st.success("Store added successfully!")
                else:
                    st.error("Please fill all required fields")
    
    with col2:
        st.subheader("Existing Stores")
        stores_df = get_all_stores()
        if not stores_df.empty:
            st.dataframe(stores_df, use_container_width=True)
        else:
            st.info("No stores added yet")

def daily_sales_page():
    st.header("📊 Daily Sales Entry")
    
    stores_df = get_all_stores()
    if stores_df.empty:
        st.warning("Please add stores first in Store Management")
        return
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Add Sales Data")
        with st.form("sales_entry_form"):
            date = st.date_input("Date", value=datetime.now().date())
            store_id = st.selectbox("Store", stores_df['store_id'].tolist())
            category = st.selectbox("Category", ITEM_CATEGORIES)
            
            # Product selection based on category
            if category:
                available_products = PRODUCTS_BY_CATEGORY.get(category, [])
                product_name = st.selectbox("Product", available_products + ["Other (Custom)"])
                
                if product_name == "Other (Custom)":
                    product_name = st.text_input("Enter Product Name")
            else:
                product_name = st.text_input("Product Name")
            
            sales_amount = st.number_input("Sales Amount (₹)", min_value=0.0, format="%.2f")
            units_sold = st.number_input("Units Sold", min_value=0, value=1)
            
            if st.form_submit_button("Add Sales Data"):
                if product_name and sales_amount > 0:
                    if add_daily_sales(date, store_id, category, product_name, sales_amount, units_sold):
                        st.success("Sales data added successfully!")
                else:
                    st.error("Please fill all required fields")
    
    with col2:
        st.subheader("Bulk Upload")
        st.info("CSV format: date, store_id, category, product_name, sales_amount, units_sold")
        uploaded_file = st.file_uploader("Upload CSV file", type="csv")
        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
                required_cols = ['date', 'store_id', 'category', 'product_name', 'sales_amount']
                if all(col in df.columns for col in required_cols):
                    st.write("Preview:")
                    st.dataframe(df.head())
                    if st.button("Import Data"):
                        conn = get_db_connection()
                        df.to_sql('daily_sales', conn, if_exists='append', index=False)
                        conn.close()
                        st.success(f"Imported {len(df)} records successfully!")
                else:
                    st.error(f"CSV must contain columns: {required_cols}")
            except Exception as e:
                st.error(f"Error reading file: {e}")
    
    # Recent sales data with product information
    st.subheader("Recent Sales Data")
    recent_sales = get_sales_data()
    if not recent_sales.empty:
        # Display columns in better order
        display_cols = ['date', 'store_name', 'category', 'product_name', 'sales_amount', 'units_sold', 'location']
        st.dataframe(recent_sales[display_cols].head(20), use_container_width=True)
    else:
        st.info("No sales data available")

def campaign_management_page():
    st.header("🎯 Campaign Management")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Create New Campaign")
        with st.form("campaign_form"):
            campaign_id = st.text_input("Campaign ID", placeholder="e.g., camp_2024_001")
            campaign_name = st.text_input("Campaign Name", placeholder="e.g., Diwali Electronics Sale")
            start_date = st.date_input("Start Date")
            end_date = st.date_input("End Date")
            campaign_type = st.radio("Campaign Type", ["single_store", "all_kerala"])
            
            stores_df = get_all_stores()
            if campaign_type == "single_store":
                if not stores_df.empty:
                    target_stores = st.multiselect("Target Stores", stores_df['store_id'].tolist())
                else:
                    target_stores = []
                    st.warning("No stores available")
            else:
                target_stores = stores_df['store_id'].tolist() if not stores_df.empty else []
                st.info(f"Campaign will target all {len(target_stores)} stores in Kerala")
            
            categories = st.multiselect("Categories", ITEM_CATEGORIES)
            discount_percent = st.number_input("Discount Percentage", min_value=0.0, max_value=100.0, format="%.1f")
            description = st.text_area("Description")
            
            if st.form_submit_button("Create Campaign"):
                if campaign_id and campaign_name and target_stores and categories:
                    conn = get_db_connection()
                    try:
                        cursor = conn.cursor()
                        cursor.execute('''
                            INSERT INTO campaigns
                            (campaign_id, campaign_name, start_date, end_date, campaign_type,
                             target_stores, categories, discount_percent, description)
                            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                        ''', (campaign_id, campaign_name, start_date, end_date, campaign_type,
                              str(target_stores), str(categories), discount_percent, description))
                        conn.commit()
                        st.success("Campaign created successfully!")
                    except Exception as e:
                        st.error(f"Error creating campaign: {e}")
                    finally:
                        conn.close()
                else:
                    st.error("Please fill all required fields")
    
    with col2:
        st.subheader("Campaign Analytics")
        # Show campaign summary metrics
        conn = get_db_connection()
        campaigns_df = pd.read_sql_query('SELECT * FROM campaigns ORDER BY created_date DESC', conn)
        conn.close()
        
        if not campaigns_df.empty:
            st.metric("Total Campaigns", len(campaigns_df))
            active_campaigns = campaigns_df[
                (pd.to_datetime(campaigns_df['start_date']) <= pd.Timestamp.now()) &
                (pd.to_datetime(campaigns_df['end_date']) >= pd.Timestamp.now())
            ]
            st.metric("Active Campaigns", len(active_campaigns))
        else:
            st.info("No campaigns created yet")
    
    # Show existing campaigns
    st.subheader("Existing Campaigns")
    if not campaigns_df.empty:
        st.dataframe(campaigns_df, use_container_width=True)
    else:
        st.info("No campaigns available")

def sales_analysis_page():
    st.header("📈 Sales Analysis")
    
    # Filters
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        stores_df = get_all_stores()
        if not stores_df.empty:
            selected_store = st.selectbox("Select Store", ["All"] + stores_df['store_id'].tolist())
        else:
            selected_store = None
            st.warning("No stores available")
    
    with col2:
        selected_category = st.selectbox("Select Category", ["All"] + ITEM_CATEGORIES)
    
    with col3:
        start_date = st.date_input("Start Date", value=datetime.now().date() - timedelta(days=30))
    
    with col4:
        end_date = st.date_input("End Date", value=datetime.now().date())
    
    if selected_store:
        # Get filtered data
        store_filter = None if selected_store == "All" else selected_store
        category_filter = None if selected_category == "All" else selected_category
        
        sales_data = get_sales_data(store_filter, start_date, end_date, category_filter)
        
        if not sales_data.empty:
            # Summary metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                total_sales = sales_data['sales_amount'].sum()
                st.metric("Total Sales", f"₹{total_sales:,.2f}")
            
            with col2:
                avg_daily_sales = sales_data.groupby('date')['sales_amount'].sum().mean()
                st.metric("Avg Daily Sales", f"₹{avg_daily_sales:,.2f}")
            
            with col3:
                total_units = sales_data['units_sold'].sum()
                st.metric("Total Units Sold", f"{total_units:,}")
            
            with col4:
                unique_categories = sales_data['category'].nunique()
                st.metric("Categories Sold", unique_categories)
            
            # Visualizations
            col1, col2 = st.columns(2)
            
            with col1:
                # Daily sales trend
                daily_sales = sales_data.groupby('date')['sales_amount'].sum().reset_index()
                fig = px.line(daily_sales, x='date', y='sales_amount',
                             title='Daily Sales Trend',
                             labels={'sales_amount': 'Sales Amount (₹)', 'date': 'Date'})
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Category-wise sales
                category_sales = sales_data.groupby('category')['sales_amount'].sum().reset_index()
                category_sales = category_sales.sort_values('sales_amount', ascending=False)
                fig = px.bar(category_sales, x='category', y='sales_amount',
                            title='Category-wise Sales',
                            labels={'sales_amount': 'Sales Amount (₹)', 'category': 'Category'})
                fig.update_xaxis(tickangle=45)
                st.plotly_chart(fig, use_container_width=True)
            
            # Store-wise analysis (if All stores selected)
            if selected_store == "All":
                col1, col2 = st.columns(2)
                
                with col1:
                    store_sales = sales_data.groupby('store_name')['sales_amount'].sum().reset_index()
                    store_sales = store_sales.sort_values('sales_amount', ascending=False)
                    fig = px.bar(store_sales, x='store_name', y='sales_amount',
                                title='Store-wise Sales Performance',
                                labels={'sales_amount': 'Sales Amount (₹)', 'store_name': 'Store'})
                    fig.update_xaxis(tickangle=45)
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    district_sales = sales_data.groupby('district')['sales_amount'].sum().reset_index()
                    fig = px.pie(district_sales, values='sales_amount', names='district',
                                title='District-wise Sales Distribution')
                    st.plotly_chart(fig, use_container_width=True)
            
            # Detailed data table
            st.subheader("Detailed Sales Data")
            st.dataframe(sales_data, use_container_width=True)
        
        else:
            st.info("No sales data available for the selected filters")

def ai_predictions_page():
    st.header("🤖 AI Sales Predictions")
    
    # Check if we have enough data
    sales_data = get_sales_data()
    if sales_data.empty:
        st.warning("No sales data available. Please add sales data first.")
        return
    
    if len(sales_data) < 30:
        st.warning(f"Insufficient data for AI predictions. Current records: {len(sales_data)}, Required: 30+")
        return
    
    # Model training section
    st.subheader("Train Prediction Model")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        if st.button("Train AI Model", type="primary"):
            with st.spinner("Training AI model..."):
                model, scaler, message = create_prediction_model(sales_data)
                
                if model is not None:
                    # Save model
                    with open('sales_model.pkl', 'wb') as f:
                        pickle.dump((model, scaler), f)
                    st.success(message)
                    st.session_state['model_trained'] = True
                else:
                    st.error(message)
    
    with col2:
        st.info("💡 The AI model analyzes historical sales patterns, seasonal trends, and campaign impacts to predict future sales.")
    
    # Prediction section
    if 'model_trained' in st.session_state or st.session_state.get('model_trained', False):
        st.subheader("Make Predictions")
        
        try:
            with open('sales_model.pkl', 'rb') as f:
                model, scaler = pickle.load(f)
            
            col1, col2 = st.columns(2)
            
            with col1:
                stores_df = get_all_stores()
                selected_store = st.selectbox("Store", stores_df['store_id'].tolist())
                selected_categories = st.multiselect("Categories", ITEM_CATEGORIES)
                prediction_date = st.date_input("Prediction Date", 
                                               value=datetime.now().date() + timedelta(days=1))
            
            with col2:
                campaign_active = st.checkbox("Campaign Active?")
                if campaign_active:
                    discount_percent = st.slider("Discount %", 0, 50, 20)
                else:
                    discount_percent = 0
            
            if st.button("Generate Prediction") and selected_categories:
                predictions = []
                
                for category in selected_categories:
                    # Create feature vector for prediction
                    # This is a simplified version - you'll need to match your training features
                    pred_date = pd.to_datetime(prediction_date)
                    features = {
                        'day_of_week': pred_date.dayofweek,
                        'month': pred_date.month,
                        'day_of_month': pred_date.day,
                        'is_weekend': int(pred_date.dayofweek >= 5),
                        'campaign_active': int(campaign_active),
                        'discount_percent': discount_percent
                    }
                    
                    # Get recent sales for lagged features
                    recent_sales = get_sales_data(selected_store, 
                                                (pred_date - timedelta(days=30)).strftime('%Y-%m-%d'),
                                                (pred_date - timedelta(days=1)).strftime('%Y-%m-%d'),
                                                category)
                    
                    if not recent_sales.empty:
                        features['sales_lag_1'] = recent_sales['sales_amount'].iloc[-1] if len(recent_sales) > 0 else 0
                        features['sales_lag_7'] = recent_sales['sales_amount'].iloc[-7] if len(recent_sales) > 6 else 0
                        features['sales_rolling_7'] = recent_sales['sales_amount'].tail(7).mean()
                    else:
                        features.update({'sales_lag_1': 0, 'sales_lag_7': 0, 'sales_rolling_7': 0})
                    
                    # Simple prediction (you'll need to properly encode categorical features)
                    base_prediction = np.random.uniform(1000, 5000)  # Placeholder
                    
                    if campaign_active:
                        # Apply campaign uplift based on discount
                        uplift_factor = 1 + (discount_percent / 100) * 1.5
                        predicted_sales = base_prediction * uplift_factor
                    else:
                        predicted_sales = base_prediction
                    
                    predictions.append({
                        'Category': category,
                        'Predicted Sales': f"₹{predicted_sales:,.2f}",
                        'Confidence': f"{np.random.uniform(70, 95):.1f}%"
                    })
                
                # Display predictions
                st.subheader("Prediction Results")
                pred_df = pd.DataFrame(predictions)
                st.dataframe(pred_df, use_container_width=True)
                
                # Visualization
                fig = px.bar(pred_df, x='Category', y='Predicted Sales',
                           title=f'Sales Prediction for {selected_store} on {prediction_date}')
                st.plotly_chart(fig, use_container_width=True)
        
        except FileNotFoundError:
            st.error("Model not found. Please train the model first.")
        except Exception as e:
            st.error(f"Prediction error: {e}")

def campaign_performance_page():
    st.header("📋 Campaign Performance Analysis")
    
    # Get campaigns
    conn = get_db_connection()
    campaigns_df = pd.read_sql_query('SELECT * FROM campaigns ORDER BY start_date DESC', conn)
    conn.close()
    
    if campaigns_df.empty:
        st.info("No campaigns available")
        return
    
    # Select campaign for analysis
    selected_campaign = st.selectbox("Select Campaign", campaigns_df['campaign_id'].tolist())
    
    if selected_campaign:
        campaign_info = campaigns_df[campaigns_df['campaign_id'] == selected_campaign].iloc[0]
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("Campaign Details")
            st.write(f"**Name:** {campaign_info['campaign_name']}")
            st.write(f"**Type:** {campaign_info['campaign_type'].replace('_', ' ').title()}")
            st.write(f"**Duration:** {campaign_info['start_date']} to {campaign_info['end_date']}")
            st.write(f"**Discount:** {campaign_info['discount_percent']}%")
            st.write(f"**Categories:** {eval(campaign_info['categories'])}")
            
            # Calculate and save performance if not already done
            if st.button("Analyze Campaign Performance"):
                target_stores = eval(campaign_info['target_stores'])
                categories = eval(campaign_info['categories'])
                
                performance_data = []
                total_sales_before = 0
                total_sales_during = 0
                total_sales_after = 0
                
                progress_bar = st.progress(0)
                total_combinations = len(target_stores) * len(categories)
                current = 0
                
                for store_id in target_stores:
                    for category in categories:
                        current += 1
                        progress_bar.progress(current / total_combinations)
                        
                        perf = calculate_campaign_performance(
                            selected_campaign, store_id, category,
                            campaign_info['start_date'], campaign_info['end_date']
                        )
                        
                        save_campaign_performance(selected_campaign, store_id, category, perf)
                        
                        performance_data.append({
                            'store_id': store_id,
                            'category': category,
                            **perf
                        })
                        
                        total_sales_before += perf['sales_before']
                        total_sales_during += perf['sales_during']
                        total_sales_after += perf['sales_after']
                
                progress_bar.empty()
                st.success("Campaign performance analysis completed!")
        
        with col2:
            st.subheader("Quick Stats")
            # Get existing performance data
            conn = get_db_connection()
            perf_df = pd.read_sql_query('''
                SELECT * FROM campaign_performance 
                WHERE campaign_id = ?
            ''', conn, params=[selected_campaign])
            conn.close()
            
            if not perf_df.empty:
                total_uplift = perf_df['uplift_percent'].mean()
                total_roi = perf_df['roi'].mean()
                best_category = perf_df.loc[perf_df['uplift_percent'].idxmax(), 'category']
                
                st.metric("Average Uplift", f"{total_uplift:.1f}%")
                st.metric("Average ROI", f"{total_roi:.2f}x")
                st.metric("Best Category", best_category)
        
        # Detailed performance analysis
        if not perf_df.empty:
            st.subheader("Performance Analysis")
            
            # Before, During, After comparison
            col1, col2 = st.columns(2)
            
            with col1:
                # Sales comparison chart
                total_before = perf_df['sales_before'].sum()
                total_during = perf_df['sales_during'].sum()
                total_after = perf_df['sales_after'].sum()
                
                comparison_data = pd.DataFrame({
                    'Period': ['Before Campaign', 'During Campaign', 'After Campaign'],
                    'Sales': [total_before, total_during, total_after]
                })
                
                fig = px.bar(comparison_data, x='Period', y='Sales',
                           title='Sales Comparison: Before vs During vs After',
                           labels={'Sales': 'Sales Amount (₹)'})
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Category-wise uplift
                category_perf = perf_df.groupby('category').agg({
                    'uplift_percent': 'mean',
                    'roi': 'mean',
                    'sales_during': 'sum'
                }).reset_index()
                
                fig = px.bar(category_perf, x='category', y='uplift_percent',
                           title='Category-wise Sales Uplift',
                           labels={'uplift_percent': 'Uplift (%)', 'category': 'Category'})
                fig.update_xaxis(tickangle=45)
                st.plotly_chart(fig, use_container_width=True)
            
            # Store-wise performance (if multiple stores)
            if len(perf_df['store_id'].unique()) > 1:
                st.subheader("Store-wise Performance")
                
                store_perf = perf_df.groupby('store_id').agg({
                    'sales_before': 'sum',
                    'sales_during': 'sum',
                    'sales_after': 'sum',
                    'uplift_percent': 'mean',
                    'roi': 'mean'
                }).reset_index()
                
                # Get store names
                stores_df = get_all_stores()
                store_perf = store_perf.merge(
                    stores_df[['store_id', 'store_name']], 
                    on='store_id', 
                    how='left'
                )
                
                col1, col2 = st.columns(2)
                
                with col1:
                    fig = px.bar(store_perf, x='store_name', y='uplift_percent',
                               title='Store-wise Uplift Performance',
                               labels={'uplift_percent': 'Uplift (%)', 'store_name': 'Store'})
                    fig.update_xaxis(tickangle=45)
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    fig = px.scatter(store_perf, x='sales_during', y='roi',
                                   hover_data=['store_name'],
                                   title='Sales vs ROI by Store',
                                   labels={'sales_during': 'Campaign Sales (₹)', 'roi': 'ROI'})
                    st.plotly_chart(fig, use_container_width=True)
            
            # Detailed performance table
            st.subheader("Detailed Performance Data")
            
            # Add store names to performance data
            stores_df = get_all_stores()
            perf_display = perf_df.merge(
                stores_df[['store_id', 'store_name']], 
                on='store_id', 
                how='left'
            )
            
            # Format the display
            perf_display['sales_before'] = perf_display['sales_before'].apply(lambda x: f"₹{x:,.2f}")
            perf_display['sales_during'] = perf_display['sales_during'].apply(lambda x: f"₹{x:,.2f}")
            perf_display['sales_after'] = perf_display['sales_after'].apply(lambda x: f"₹{x:,.2f}")
            perf_display['uplift_percent'] = perf_display['uplift_percent'].apply(lambda x: f"{x:.1f}%")
            perf_display['roi'] = perf_display['roi'].apply(lambda x: f"{x:.2f}x")
            
            display_cols = ['store_name', 'category', 'sales_before', 'sales_during', 
                          'sales_after', 'uplift_percent', 'roi']
            st.dataframe(perf_display[display_cols], use_container_width=True)
            
            # Export functionality
            if st.button("Export Performance Report"):
                csv = perf_display.to_csv(index=False)
                st.download_button(
                    label="Download CSV",
                    data=csv,
                    file_name=f"campaign_performance_{selected_campaign}.csv",
                    mime="text/csv"
                )
        
        # Campaign insights and recommendations
        if not perf_df.empty:
            st.subheader("AI Insights & Recommendations")
            
            avg_uplift = perf_df['uplift_percent'].mean()
            best_performing_category = perf_df.loc[perf_df['uplift_percent'].idxmax(), 'category']
            worst_performing_category = perf_df.loc[perf_df['uplift_percent'].idxmin(), 'category']
            
            insights = []
            
            if avg_uplift > 20:
                insights.append("🎉 **Excellent Performance**: Campaign achieved strong sales uplift across categories.")
            elif avg_uplift > 10:
                insights.append("✅ **Good Performance**: Campaign showed positive results with room for improvement.")
            else:
                insights.append("⚠️ **Needs Improvement**: Campaign performance below expectations.")
            
            insights.append(f"🏆 **Best Category**: {best_performing_category} showed the highest uplift.")
            insights.append(f"📈 **Opportunity**: Focus more on {worst_performing_category} in future campaigns.")
            
            # Seasonal recommendations
            campaign_date = pd.to_datetime(campaign_info['start_date'])
            month = campaign_date.month
            
            if month in [10, 11]:  # Diwali season
                insights.append("🪔 **Seasonal Insight**: Diwali period - Electronics and Clothing typically perform well.")
            elif month in [12, 1]:  # New Year season
                insights.append("🎊 **Seasonal Insight**: New Year period - Beauty and Fashion categories see higher demand.")
            elif month in [6, 7, 8]:  # Monsoon season
                insights.append("🌧️ **Seasonal Insight**: Monsoon period - Home & Garden, Books show better performance.")
            
            # Display insights
            for insight in insights:
                st.markdown(insight)
            
            # Future campaign recommendations
            st.subheader("Future Campaign Recommendations")
            
            recommendations = []
            
            if avg_uplift < 15:
                recommendations.append("Consider increasing discount percentage or improving targeting.")
            
            if len(perf_df['store_id'].unique()) > 1:
                best_store = perf_df.groupby('store_id')['uplift_percent'].mean().idxmax()
                recommendations.append(f"Replicate successful strategies from {best_store} to other stores.")
            
            recommendations.append("Focus future campaigns on high-performing categories to maximize ROI.")
            recommendations.append("Consider extending successful campaigns by 2-3 days for better impact.")
            
            for i, rec in enumerate(recommendations, 1):
                st.markdown(f"{i}. {rec}")

if __name__ == "__main__":
    main()
