import pandas as pd
import random
from faker import Faker
import os

fake = Faker()

# Research fields and subfields (updated for publication keywords)
fields = {
    "Computer Science": ["AI", "Machine Learning", "NLP", "Cybersecurity", "Computer Vision"],
    "Biology": ["Genetics", "Neuroscience", "Microbiology", "Ecology"],
    "Physics": ["Quantum", "Astrophysics", "Condensed Matter", "Particle Physics"],
    "Engineering": ["Electrical", "Mechanical", "Civil", "Biomedical"],
    "Mathematics": ["Statistics", "Applied Math", "Pure Math", "Computational"]
}

# --- Generate Researchers (100 entries) ---
print("👩‍🔬 Generating researchers...")
researchers = []
for i in range(100):
    field = random.choice(list(fields.keys()))
    researchers.append({
        "researcher_id": f"R{i:03d}",
        "name": fake.name(),
        "department": field,
        "email": fake.email(),
        "research_interests": ", ".join(random.sample(fields[field], 3))
    })
researchers_df = pd.DataFrame(researchers)

# --- Generate Publications (200 entries with REALISTIC author links) ---
print("📚 Generating publications with author collaborations...")
publications = []
for i in range(200):
    # Select 1-4 authors from the SAME DEPARTMENT (more realistic collaborations)
    department = random.choice(list(fields.keys()))
    potential_authors = researchers_df[researchers_df['department'] == department]['name'].tolist()
    authors = random.sample(potential_authors, min(random.randint(1, 4), len(potential_authors)))
    
    # Generate publication data
    publications.append({
        "publication_id": f"P{i:04d}",
        "title": fake.sentence(nb_words=6).replace('.', ''),  # Remove period for cleaner titles
        "abstract": fake.paragraph(nb_sentences=3),
        "authors": ", ".join(authors),
        "year": random.randint(2015, 2023),
        "keywords": ", ".join(random.sample(fields[department], 2))  # 2 keywords per paper
    })

publications_df = pd.DataFrame(publications)

# --- Save Files ---
output_dir = "research_data"
os.makedirs(output_dir, exist_ok=True)

researchers_df.to_csv(f"{output_dir}/researchers.csv", index=False)
publications_df.to_csv(f"{output_dir}/publications.csv", index=False)

print(f"\n✅ Successfully generated:\n- {len(researchers_df)} researchers\n- {len(publications_df)} publications")
print(f"📂 Files saved to: {os.path.abspath(output_dir)}")