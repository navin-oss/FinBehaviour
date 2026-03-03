import random
import json
from datetime import datetime, timedelta

# Templates for realistic Indian financial social posts
TEMPLATES = {
    "Spending": [
        "Bought {} today! 😍",
        "Just spent ₹{} on {} - worth it!",
        "New {} arrived! Unboxing soon 📦",
        "EMI started for my new {} - monthly ₹{}",
        "Treated myself to {} after promotion! 🎉",
        "Splurged ₹{} on {} this weekend, no regrets!",
        "Finally got {} on Flipkart sale - saved ₹{}!"
    ],
    "Investment": [
        "Invested ₹{} in {} today 📈",
        "Started SIP: ₹{}/month in {} - future me will thank me",
        "Portfolio update: Added {} worth ₹{}",
        "Learning about {} investing. Small start: ₹{}",
        "Rebalanced portfolio: Shifted ₹{} to {}",
        "Dividend received ₹{} from {}! Passive income 💪",
        "PPF matured at ₹{}, reinvesting in {}"
    ],
    "Loan": [
        "Loan of ₹{} approved for {}! Dreams coming true 🎉",
        "Paid EMI of ₹{} for my {} loan - {} months to go",
        "Applied for personal loan. Fingers crossed 🤞",
        "Closed my {} loan today! Freedom feels amazing 🙌",
        "Pre-approved for {} loan of ₹{}. Tempting...",
        "Home loan interest rate dropped to {}% - refinancing!",
        "Education loan EMI ₹{} started for {}"
    ],
    "Savings": [
        "Transferred ₹{} to savings account 💰",
        "Saved ₹{} this month! Goal: ₹{} by Diwali",
        "Emergency fund updated: ₹{} saved - 6 months covered ✅",
        "Cut expenses, saved ₹{} extra this week",
        "Auto-debit set: ₹{} to FD every month",
        "No-spend challenge day {}: saved ₹{} already!",
        "Reached ₹{} in my {} savings goal! 🎯"
    ],
    "Risk": [
        "All in on {} crypto! 🚀 To the moon!",
        "Lost ₹{} in {} trading... lesson learned 😅",
        "Borrowed ₹{} to invest in {}. High risk, high reward?",
        "Fantasy league winnings: ₹{}! Trying my luck again",
        "Options trading: Made ₹{} today. Feels like gambling but profitable?",
        "Put ₹{} in {} without research. YOLO! 🎲",
        "Leveraged ₹{} on {} futures. Let's see what happens..."
    ]
}

ITEMS = {
    "Spending": ["iPhone 15", "MacBook Pro", "Nike Air Max", "Fossil watch", "Sony WH-1000XM5", "Zara outfit", "PS5", "iPad Air"],
    "Investment": ["Nifty 50 index fund", "SBI Bluechip fund", "Gold ETF", "NPS Tier-I", "Axis long term equity", "HDFC Mid-Cap", "Sovereign Gold Bond"],
    "Loan": ["Honda Activa", "MBA education", "1BHK flat", "personal emergency", "Maruti Swift", "laptop", "wedding expenses"],
    "Savings": ["SBI FD", "PPF account", "emergency fund", "Goa trip fund", "child education corpus", "retirement fund", "house down payment"],
    "Risk": ["Shiba Inu", "penny stocks", "intraday options", "Dream11", "lottery scratch cards", "meme coins", "leveraged crypto"]
}

AMOUNTS = [500, 1000, 2500, 5000, 10000, 15000, 25000, 50000, 75000, 100000, 200000]

def generate_post(category):
    template = random.choice(TEMPLATES[category])
    item = random.choice(ITEMS[category])
    amount = random.choice(AMOUNTS)
    amount2 = random.choice(AMOUNTS)
    
    placeholders = template.count("{}")
    if placeholders == 1:
        return template.format(item)
    elif placeholders == 2:
        return template.format(amount, item)
    elif placeholders == 3:
        return template.format(amount, item, amount2)
    return template

# Generate balanced dataset with timestamps
synthetic_data = []
base_date = datetime.now() - timedelta(days=30)

for category in TEMPLATES.keys():
    for i in range(15):  # 15 posts per category = 75 total
        post_date = base_date + timedelta(days=random.randint(0, 30), hours=random.randint(6, 23))
        synthetic_data.append({
            "text": generate_post(category),
            "true_label": category,
            "timestamp": post_date.isoformat(),
            "source": "synthetic_generator_v2"
        })

# Shuffle for realism
random.shuffle(synthetic_data)

# Save to JSON
with open("synthetic_posts.json", "w", encoding="utf-8") as f:
    json.dump(synthetic_data, f, indent=2, ensure_ascii=False)

print(f"✅ Generated {len(synthetic_data)} synthetic posts in 'synthetic_posts.json'")
print(f"📅 Date range: {base_date.strftime('%Y-%m-%d')} to {datetime.now().strftime('%Y-%m-%d')}")
print("💡 Tip: Upload this JSON file in the app for demo data")
