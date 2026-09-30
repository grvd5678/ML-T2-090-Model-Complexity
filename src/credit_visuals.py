import matplotlib.pyplot as plt

# Our hard-earned data from the terminal!
labels = ['Logistic Regression', 'Random Forest']
recall_scores = [0.54, 0.44] # How many bad loans we actually caught
times = [0.0165, 0.2408]     # How fast they ran

# Setting up our classic side-by-side view
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

# Chart 1: Recall (Catching Bad Loans)
ax1.bar(labels, recall_scores, color=['#7f8c8d', '#e74c3c'])
ax1.set_ylabel('Recall Score (Bad Credit)')
ax1.set_title('Catching Bad Loans (Higher is Better)')
ax1.set_ylim(0.0, 1.0) 

# Chart 2: Engineering Time
ax2.bar(labels, times, color=['#7f8c8d', '#e74c3c'])
ax2.set_ylabel('Time (seconds)')
ax2.set_title('Engineering Cost (Lower is Better)')

plt.tight_layout()

# Save it traditionally for the final paper
plt.savefig('docs/credit_model_tradeoff.png')
print("Chart successfully saved to docs/credit_model_tradeoff.png! 💅")

# Pop it open!
plt.show()