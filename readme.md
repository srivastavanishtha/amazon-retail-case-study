**Problem Statement:**

As a data scientist at an e-commerce company, you've been tasked with improving the product recommendation system and pricing strategy. The company wants to leverage its vast database of product information, customer reviews, and pricing data to achieve the following goals:

1. Develop a simple but effective product recommendation system.
2. Analyze customer sentiment to understand product reception.
3. Optimize pricing strategies based on product features and customer feedback.

You have access to a dataset containing information about various products, including their descriptions, prices, discounts, and customer reviews. Your challenge is to build an end-to-end data analysis pipeline that addresses these goals and provides actionable insights for the business.

**Dataset:**
The dataset includes the following key features:
- product_id, product_name, category
- discounted_price, actual_price, discount_percentage
- rating, rating_count
- about_product (description)
- review_title, review_content
- user_id, user_name

**Tasks:**

1. **Data Preprocessing and Exploration:**
   - Load and clean the dataset
   - Perform basic exploratory data analysis
   - Handle missing values and outliers

2. **Feature Engineering:**
   - Extract relevant features from the text data (product descriptions and reviews)
   - Create numerical features for the recommendation system and pricing analysis

3. **Sentiment Analysis:**
   - Implement a simple sentiment analysis model on the review data
   - Analyze the relationship between sentiment and product ratings

4. **Product Recommendation System:**
   - Develop a basic collaborative filtering recommendation system
   - Evaluate the system's performance using appropriate metrics

5. **Pricing Analysis:**
   - Analyze the relationship between product features, sentiment, and pricing
   - Build a simple model to suggest optimal discount percentages

6. **Insights and Recommendations:**
   - Summarize key findings from the analysis
   - Provide actionable recommendations for the business

The solution should be implemented in Python and designed to run efficiently on Google Colab. The entire analysis pipeline should be executable within a 20-minute timeframe.