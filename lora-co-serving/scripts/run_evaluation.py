# scripts/run_evaluation.py

import logging
import sys
import time
import threading
import uuid
from dataclasses import asdict, dataclass
from typing import List, Dict, Any, Optional
import numpy as np
import json
from datetime import datetime

# Add project root to path
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from utils.logger_config import setup_logger
from utils.config_loader import load_config, SystemConfig
from core.models import load_base_model
import torch

from core.engine import InferenceEngine
from managers.queue_manager import QueueManager
from managers.adapter_manager import LoRAAdapterManager
from managers.job_manager import ActiveTrainingJobManager, JobStatus
from prioritization.round_robin import RoundRobinStrategy
from prioritization.least_progress_first import LeastProgressFirstStrategy
from prioritization.stagnation_aware import ForwardStagnationAwareStrategy, ReverseStagnationAwareStrategy
from core.controller import MainController
from core.training import setup_lora_training

@dataclass
class ScenarioEvent:
    delay_seconds: float
    event_type: str # 'TRAIN_JOB', 'INFERENCE_REQ'
    details: Dict[str, Any]


def scenario_submitter(queue_manager: QueueManager, scenario: List[ScenarioEvent]):
    """Submits events from the scenario to the queue manager with delays."""
    logger = logging.getLogger("ScenarioSubmitter")
    logger.info(f"Starting scenario submission. Total events: {len(scenario)}")
    start_time = time.time()
    for event in scenario:
        current_time = time.time()
        target_time = start_time + event.delay_seconds
        sleep_duration = max(0, target_time - current_time)
        if sleep_duration > 0:
            logger.debug(f"Sleeping for {sleep_duration:.2f} seconds before submitting next event.")
            time.sleep(sleep_duration)
        
        if event.event_type == 'TRAIN_JOB':
            # Add submission time for potential analysis later if needed
            event.details['submission_time'] = time.time()
            logger.info(f"Submitting training job: {event.details.get('job_id', 'N/A')}")
            queue_manager.add_training_job(event.details)
        elif event.event_type == 'INFERENCE_REQ':
            # Ensure unique request IDs for inference
            event.details['request_id'] = f"inf_{uuid.uuid4().hex[:8]}"
            # Add submission time for latency calculation
            event.details['submission_time'] = time.time()
            logger.info(f"Submitting inference request: {event.details['request_id']} at {event.details['submission_time']:.2f}")
            queue_manager.add_inference_request(event.details)
        else:
            logger.warning(f"Unknown event type in scenario: {event.event_type}")
            
    logger.info("Finished submitting all scenario events.")


def define_scenario_a(base_model_name: str) -> List[ScenarioEvent]:
    """Defines a mixed load scenario with longer text samples."""
    
    # Generate longer, more realistic dummy data
    dataset_1 = [
        "Instruction: Provide a comprehensive summary of the following text, ensuring all key arguments and supporting details are captured concisely. Text: The quick brown fox, known for its agility and speed, effortlessly jumps over the sleeping, unsuspecting lazy dog. This event occurred on a particularly bright and sunny afternoon in the expansive, green central park, witnessed by several onlookers enjoying the pleasant weather.",
        "Question: Analyze the provided text and extract the core message or main idea presented, explaining the context and significance. Answer: The central theme illustrated is the inherent contrast between agility and lethargy, symbolized by the active fox and the passive dog, observed within a peaceful, everyday park setting on a sunny day.",
        "Provide a detailed scientific explanation of photosynthesis, covering the light-dependent and light-independent reactions, the role of chlorophyll, and the overall chemical equation. Discuss its importance for plant life and the global ecosystem.",
        "Analyze the sentiment conveyed by the phrase: 'While the acting was commendable, the plot was predictable and the ending felt rushed, ultimately leaving me feeling unsatisfied with the cinematic experience, despite the initial high expectations.' Identify nuances in tone.",
        "Generate three distinct and insightful follow-up questions to the statement: 'Regular physical exercise offers numerous health benefits, including improved cardiovascular health, weight management, and enhanced mental well-being.' Consider different angles like specific types of exercise, long-term effects, or potential barriers.",
        "Correct the following sentence for grammatical errors, improving clarity and flow: 'He go to the store yesterday for buy breads milk and eggs but forget his wallet.' Explain the corrections made regarding verb tense, prepositions, and conjunctions.",
        "Explain the fundamental differences between the LoRA (Low-Rank Adaptation) technique and traditional full fine-tuning for large language models. Discuss the trade-offs in terms of parameter efficiency, computational cost, memory usage, and potential performance implications.",
        "Write a compelling and informative product description for a newly launched smartwatch. Highlight its key features (e.g., health tracking, battery life, connectivity options, unique selling points) and target audience, aiming to persuade potential customers.",
        "Summarize the following paragraph detailing the advantages and challenges of various renewable energy sources like solar, wind, and geothermal power. Ensure the summary accurately reflects the main points regarding efficiency, cost, environmental impact, and intermittency.",
        "Describe the key features and design philosophies of the Python programming language. Include aspects like readability, dynamic typing, extensive standard library, garbage collection, object-oriented capabilities, and its suitability for various domains like web development, data science, and scripting."
    ] # 10 samples total

    dataset_2 = [
        "Instruction: Translate the following extended greeting and inquiry into formal French, paying attention to appropriate politeness levels. Text: 'Good morning, sir. I hope this message finds you well. Might I inquire as to your current well-being and availability for a brief meeting later today?'",
        "Question: Calculate the result of the mathematical expression 5 plus 7, then multiply the sum by 3 and subtract 10. Explain each step. Answer: First, 5 plus 7 equals 12. Next, 12 multiplied by 3 equals 36. Finally, subtracting 10 from 36 yields the result of 26.",
        "Provide a detailed, step-by-step guide for beginners on how to bake a simple loaf of artisan sourdough bread from scratch. Include ingredient measurements, starter maintenance, mixing techniques, fermentation times, shaping, proofing, scoring, and baking instructions.",
        "Explain the core concept of recursion in computer programming. Define base cases and recursive steps. Provide a clear, practical code example (e.g., calculating factorials or traversing a tree structure) and trace its execution flow to illustrate how the stack is used.",
        "Generate three distinct and evocative titles for a science fiction novel centered around a perilous multi-generational voyage to colonize a distant exoplanet facing unforeseen cosmic dangers and internal societal conflicts.",
        "Analyze the underlying tone and implied message of the following business email communication: 'Subject: Meeting Update. Body: Just a heads-up, the project review meeting previously scheduled for 10 AM has been moved to 3 PM this afternoon due to a scheduling conflict. Please adjust your calendars accordingly.'",
        "Discuss the primary economic, social, and ethical arguments both for and against the implementation of a universal basic income (UBI) policy. Consider potential impacts on poverty reduction, labor market participation, innovation, and government finances.",
        "Write a reflective short poem capturing the distinct moods and sensory experiences associated with the transition through the four seasons: the renewal of spring, the warmth of summer, the colors of autumn, and the quietude of winter.",
        "Provide a concise yet comprehensive summary of the main plot points, central themes (like social class, marriage, and personal growth), and key character arcs in Jane Austen's novel 'Pride and Prejudice'.",
        "Explain the historical context and significance of the Turing test as proposed by Alan Turing. Discuss its role in the philosophy of artificial intelligence, its limitations, and alternative approaches to evaluating machine intelligence.",
        "Describe the complete water cycle process in detail, explaining evaporation, transpiration, condensation, precipitation, infiltration, runoff, and collection. Mention the driving forces (solar energy, gravity) and its importance for Earth's climate and ecosystems.",
        "Generate five relevant and potentially trending hashtags for a social media campaign promoting sustainable living practices, focusing on reducing plastic waste, conserving energy, and supporting ethical consumerism."
    ] # 12 samples total

    dataset_3 = [
        "Instruction: Identify and list all named entities (persons, organizations, locations) mentioned in the following news excerpt. Text: 'Cupertino-based Apple Inc. unveiled its latest iPhone model during a major press event held in California last Tuesday, with CEO Tim Cook presenting the new features.'",
        "Question: Identify the author of the renowned tragic play 'Hamlet', which explores themes of revenge, betrayal, and madness. Provide brief context about the playwright. Answer: The celebrated play 'Hamlet' was written by the English playwright William Shakespeare, considered one of the greatest writers in the English language.",
        "Generate a simple, easy-to-follow recipe for a classic spaghetti carbonara dish. Include a list of necessary ingredients (pasta, guanciale/pancetta, eggs, Pecorino Romano, black pepper), estimated preparation/cooking times, and clear, sequential instructions.",
        "Explain the fundamental principles of blockchain technology. Describe concepts like decentralization, distributed ledgers, cryptographic hashing, consensus mechanisms (e.g., Proof-of-Work, Proof-of-Stake), and immutability. Mention potential applications beyond cryptocurrencies.",
        "Write three distinct and catchy marketing slogans for a new brand of eco-friendly, plant-based household cleaning products designed to be safe for families and pets while being effective on tough grime."
    ] # 5 samples total

    scenario = [
        # --- Initial Jobs --- 
        ScenarioEvent(delay_seconds=0.1, event_type='TRAIN_JOB', details={
            "job_id": "scenA_train_1", "base_model_id": base_model_name,
            "dataset_samples": dataset_1, "max_steps": 10,
            "training_params": {"lr": 5e-5, "batch_size": 1},
            "adapter_save_path": "./adapters/scenA_train_1"
        }),
        ScenarioEvent(delay_seconds=0.5, event_type='TRAIN_JOB', details={
            "job_id": "scenA_train_2", "base_model_id": base_model_name,
            "dataset_samples": dataset_2, "max_steps": 12,
            "training_params": {"lr": 3e-5, "batch_size": 1},
            "adapter_save_path": "./adapters/scenA_train_2"
        }),
        # --- Inference Stream Starts --- 
        ScenarioEvent(delay_seconds=2.0, event_type='INFERENCE_REQ', details={
            "prompt": "What is the weather like?", "adapter_path": None
        }),
        ScenarioEvent(delay_seconds=3.0, event_type='INFERENCE_REQ', details={
            "prompt": "Tell me a short story.", "adapter_path": None 
        }),
        # --- Third Job Arrives --- 
        ScenarioEvent(delay_seconds=5.0, event_type='TRAIN_JOB', details={
            "job_id": "scenA_train_3", "base_model_id": base_model_name,
            "dataset_samples": dataset_3, "max_steps": 6,
            "training_params": {"lr": 4e-5, "batch_size": 1},
            "adapter_save_path": "./adapters/scenA_train_3"
        }),
        # --- More Inference --- 
        ScenarioEvent(delay_seconds=6.0, event_type='INFERENCE_REQ', details={
            "prompt": "Explain the concept of LoRA.", "adapter_path": None
        }),
        # Add more events as needed, potentially using trained adapters later
    ]
    return scenario


# --- Add New Scenario Definition ---
def define_scenario_b(base_model_name: str) -> List[ScenarioEvent]:
    """Defines a scenario with varying job lengths and arrival times, using longer texts."""

    # Generate unique dummy data
    dataset_b1 = [
        "Instruction: Rephrase the following sentence to convey the same meaning but using a more active voice and varied vocabulary. Sentence: 'The comprehensive status report regarding the project's progress was meticulously completed by the dedicated development team before the deadline.'",
        "Question: Calculate the precise square root of 64 and then determine if the result is a prime number. Explain your reasoning. Answer: The square root of 64 is exactly 8. The number 8 is not prime because it is divisible by 1, 2, 4, and 8 (more than just 1 and itself).",
        "Generate a clever and contextually appropriate witty response to the classic philosophical joke: 'Why did the chicken cross the road?' Consider the potential motivations of the chicken or the absurdity of the question itself.",
        "Explain the key technical differences between the Hypertext Transfer Protocol (HTTP) and its secure counterpart, HTTPS. Discuss the role of SSL/TLS encryption, port numbers, and the implications for data security and user privacy during web browsing.",
        "Write a brief but sincere thank-you note expressing gratitude for a thoughtful gift received recently. Mention the specific gift and express how much you appreciate the gesture and the sender's consideration."
    ] * (5 // 5)

    # Larger dataset for the long job with expanded base texts
    long_job_base_texts = [
        "Instruction: Elaborate extensively on the multifaceted impacts of accelerating climate change specifically on vulnerable coastal regions worldwide. Discuss rising sea levels, increased storm surge intensity, coastal erosion, saltwater intrusion into freshwater sources, and the socio-economic consequences for communities and ecosystems.",
        "Question: Describe the intricate process of natural selection as a primary mechanism of evolution, using Darwin's finches and antibiotic-resistant bacteria as illustrative examples. Explain variation, inheritance, differential survival, and reproduction. Answer: Natural selection, a cornerstone of evolutionary theory proposed by Charles Darwin, operates on existing variation within a population...",
        "Generate a highly detailed and multi-dimensional character profile for the main protagonist of a dark fantasy novel. Include physical appearance, personality traits (strengths, flaws, motivations), background history, key relationships, skills/abilities, internal conflicts, and their overarching goal within the narrative.",
        "Explain Albert Einstein's theory of special relativity and general relativity in conceptually accessible terms for a lay audience. Cover concepts like the constancy of the speed of light, time dilation, length contraction, spacetime curvature, gravity as geometry, and provide analogies where helpful.",
        "Write a comprehensive and well-structured summary analyzing the main social, economic, technological, and political effects of the Industrial Revolution in Europe and North America during the 18th and 19th centuries. Discuss urbanization, factory systems, new class structures, technological innovations, and global power shifts.",
        "Analyze the complex array of arguments surrounding nuclear energy as a power source. Discuss the pros (low carbon emissions, high power density, energy security) and cons (radioactive waste disposal, accident risks like Chernobyl/Fukushima, proliferation concerns, high initial costs) from scientific, environmental, and economic perspectives.",
        "Provide a detailed technical explanation of how Convolutional Neural Networks (CNNs) function, particularly in the context of image recognition tasks. Describe the roles of convolutional layers, pooling layers (max/average), activation functions (ReLU), fully connected layers, feature hierarchies, and the training process using backpropagation.",
        "Discuss the profound ethical implications arising from the increasing use of artificial intelligence and machine learning algorithms in critical decision-making processes across various domains, such as healthcare diagnostics, loan applications, hiring practices, autonomous vehicles, and criminal justice systems. Consider bias, fairness, transparency, accountability, and potential societal impacts.",
        "Summarize the central philosophical arguments, key concepts (such as the theory of Forms, the allegory of the cave, the ideal state, philosopher-kings), and main dialogues presented in Plato's influential work, 'The Republic'. Discuss its lasting impact on Western philosophy and political thought.",
        "Write a detailed and comparative blog post evaluating the major cloud computing platforms: Amazon Web Services (AWS), Microsoft Azure, and Google Cloud Platform (GCP). Compare their core services (compute, storage, database, networking, AI/ML), pricing models, market share, target audiences, strengths, and weaknesses for different use cases."
    ]
    dataset_b2 = [text for text in long_job_base_texts] * (40 // len(long_job_base_texts))
    # Ensure correct length if division is not exact
    if len(dataset_b2) < 40:
        dataset_b2.extend([long_job_base_texts[i % len(long_job_base_texts)] for i in range(len(dataset_b2), 40)])


    dataset_b3 = [
        "Instruction: Craft five distinct and professional subject lines for an email requesting a brief introductory meeting with a potential business partner or industry contact, ensuring clarity and a compelling reason to open the email.",
        "Question: Convert 50 degrees Celsius to Fahrenheit using the standard conversion formula (F = C * 9/5 + 32), showing the calculation steps clearly. Answer: To convert 50°C to Fahrenheit, we calculate (50 * 9/5) + 32. This simplifies to (9 * 10) + 32 = 90 + 32 = 122°F.",
        "Generate an extensive list of synonyms (at least ten) for the adjective 'important', covering various nuances such as significance, urgency, value, and influence (e.g., crucial, vital, essential, significant, paramount, consequential, critical, substantial, noteworthy, principal).",
        "Explain the fundamental concepts of object-oriented programming (OOP), including classes, objects, encapsulation, inheritance, polymorphism, and abstraction. Provide simple analogies or brief code snippets to illustrate each principle.",
        "Write a short, natural-sounding dialogue between two friends who are excitedly planning a weekend camping trip. Include discussion about location options, necessary gear, food planning, and potential activities."
    ] * (5 // 5)


    scenario = [
        # --- Job B1 (Short) Starts ---
        ScenarioEvent(delay_seconds=0.1, event_type='TRAIN_JOB', details={
            "job_id": "scenB_train_1_short", "base_model_id": base_model_name,
            "dataset_samples": dataset_b1, "max_steps": 5,
            "training_params": {"lr": 5e-5, "batch_size": 1},
            "adapter_save_path": "./adapters/scenB_train_1_short"
        }),
        # --- Job B2 (Long) Starts ---
        ScenarioEvent(delay_seconds=1.0, event_type='TRAIN_JOB', details={
            "job_id": "scenB_train_2_long", "base_model_id": base_model_name,
            "dataset_samples": dataset_b2, "max_steps": 40, # Much longer
            "training_params": {"lr": 3e-5, "batch_size": 1},
            "adapter_save_path": "./adapters/scenB_train_2_long"
        }),
        # --- Early Inference ---
        ScenarioEvent(delay_seconds=2.0, event_type='INFERENCE_REQ', details={
            "prompt": "What is the capital of Spain?", "adapter_path": None
        }),
        # --- Job B3 (Short, Late) Arrives during B2's run ---
        ScenarioEvent(delay_seconds=10.0, event_type='TRAIN_JOB', details={
            "job_id": "scenB_train_3_short_late", "base_model_id": base_model_name,
            "dataset_samples": dataset_b3, "max_steps": 5,
            "training_params": {"lr": 4e-5, "batch_size": 1},
            "adapter_save_path": "./adapters/scenB_train_3_short_late"
        }),
        # --- Mid-run Inference ---
        ScenarioEvent(delay_seconds=15.0, event_type='INFERENCE_REQ', details={
            "prompt": "Summarize the plot of Hamlet.", "adapter_path": None
        }),
         # --- Later Inference ---
        ScenarioEvent(delay_seconds=25.0, event_type='INFERENCE_REQ', details={
            "prompt": "What are the main differences between Python 2 and 3?", "adapter_path": None
        }),
    ]
    return scenario
# --- End New Scenario Definition ---

# --- Add New Scenario Definition ---
def define_scenario_c(base_model_name: str) -> List[ScenarioEvent]:
    """Defines Scenario C with higher concurrency, using longer texts."""

    # Generate longer unique dummy data for 4 jobs
    dataset_c1 = [
        "Instruction: Identify the main verb and its tense in the following complex sentence. Sentence: 'While the diligent cat patiently sleeps soundly in the warm afternoon sun, the birds outside are chirping merrily.'",
        "Question: What is the complete chemical formula for water, and what types of bonds hold the molecule together? Explain briefly. Answer: The chemical formula for water is H2O. The oxygen atom forms covalent bonds with two hydrogen atoms.",
        "Generate three different simple greetings suitable for starting a casual conversation with a stranger in an informal setting, varying in tone.",
        "What does the acronym CPU stand for in the context of computer hardware, and what is its primary function within a computer system?",
        "Write three descriptive sentences capturing the different visual elements and mood of a vibrant, multi-colored sunset over a calm ocean horizon."
    ] * (5 // 5)

    dataset_c2 = [
        "Instruction: Read the following short passage about recent advancements in artificial intelligence and provide a concise, accurate summary capturing the main points discussed. Passage: [Insert a relevant 100-word passage about AI advancements here].",
        "Question: List three distinct types of renewable energy sources currently being utilized globally and briefly describe how each generates power. Answer: 1. Solar Power (uses photovoltaic cells to convert sunlight). 2. Wind Power (uses turbines to harness wind energy). 3. Hydroelectric Power (uses water flow through dams).",
        "Generate a thought-provoking open-ended question regarding the future ethical challenges and societal impact of increasingly sophisticated machine learning models.",
        "Explain the fundamental difference between RAM (Random Access Memory) and ROM (Read-Only Memory) in a computer system, focusing on their purpose, volatility, speed, and typical usage.",
        "Write a balanced short product review for a recently read novel, highlighting both its strengths (e.g., compelling characters, engaging plot) and weaknesses (e.g., slow pacing, predictable ending).",
        "What is the primary purpose of a network firewall in cybersecurity, and what are the main types of threats it aims to protect against?",
        "Generate a simple, well-commented Python function that takes two numerical arguments and returns their sum, illustrating basic function definition and return statements.",
        "Describe the essential role and key functions of a Database Management System (DBMS) in storing, retrieving, managing, and ensuring the integrity of large volumes of data.",
        "What are the three primary colors in the additive color model (used for light/screens), and what happens when they are combined in equal proportions?",
        "Construct a grammatically correct and meaningful sentence that effectively uses the word 'ubiquitous' to describe the prevalence of smartphones in modern society.",
        "Explain Moore's Law, as originally observed by Gordon Moore, regarding the trend in transistor density on integrated circuits, and discuss its historical impact and current relevance.",
        "Generate a list of five creative and engaging blog post titles centered around budget-friendly international travel tips and destination ideas.",
        "What is the primary biological function of the mitochondria within eukaryotic cells, often referred to as the 'powerhouse' of the cell?",
        "Summarize the basic economic concept of supply and demand, explaining how the interaction between the availability of a product and the desire for it influences its market price.",
        "Write a traditional 5-7-5 syllable haiku capturing a specific moment or image observed in the natural world during the autumn season."
    ] * (15 // 15)

    dataset_c3 = [
        "Instruction: Identify the misspelling in the following sentence and provide the correct spelling. Sentence: 'The committee needs to accomodate the diverse needs of all participants to ensure a successful event.'",
        "Question: What is the official capital city of the Commonwealth of Australia, and in which territory is it located? Answer: The capital of Australia is Canberra, located within the Australian Capital Territory (ACT).",
        "Generate a polite and considerate refusal message suitable for declining a social invitation to an event you are unable to attend, expressing regret and appreciation.",
        "Explain what an API (Application Programming Interface) is in software development. Describe its purpose in enabling communication and data exchange between different software systems or components, using a simple analogy.",
        "Write an evocative and descriptive caption suitable for posting alongside a photograph showcasing a majestic snow-capped mountain landscape under a clear blue sky.",
        "What are the five main hardware components typically found inside a modern personal computer case? Briefly describe the function of each (e.g., CPU, RAM, Motherboard, Storage Drive, Power Supply).",
        "Generate a simple code example in Python demonstrating the use of a 'for' loop to iterate over a list of numbers and print each number.",
        "List three distinct health benefits associated with regularly incorporating a variety of fresh vegetables into one's diet, such as providing essential vitamins or improving digestion."
    ] * (8 // 8)

    # Long job dataset with expanded base texts
    long_job_c_base = [
        "Instruction: Write a comprehensive and detailed technical explanation of the Low-Rank Adaptation (LoRA) fine-tuning technique. Cover its theoretical underpinnings based on low-rank matrix decomposition, the mathematical formulation involving matrices A and B, the role of the scaling factor alpha, practical implementation details within transformer architectures (targeting specific modules like attention layers), and discuss its advantages (parameter efficiency, reduced memory, faster training, modularity) and potential limitations compared to full fine-tuning.",
        "Question: Provide a comprehensive overview and detailed discussion of the major ongoing challenges and open research problems within the field of Natural Language Processing (NLP). Address issues such as handling ambiguity (lexical, syntactic, semantic), context dependency and understanding, common sense reasoning, scalability for massive datasets and models, data scarcity for low-resource languages, robustness against adversarial attacks, bias and fairness in language models, and the explainability of complex NLP systems.",
        "Generate a multi-paragraph fictional narrative (at least 500 words) based on the following creative prompt: 'An international team of scientists drilling deep into the Antarctic ice sheet makes a startling discovery far beneath the surface – not of ancient microbes, but of a perfectly preserved, technologically advanced structure of unknown origin, emitting a faint energy signature.' Develop the initial discovery scene, character reactions, and the immediate scientific and geopolitical implications.",
        "Explain the core architecture and the adversarial training process of Generative Adversarial Networks (GANs) in detail. Describe the roles and objectives of the Generator network (mapping latent space to data space) and the Discriminator network (distinguishing real vs. fake data). Cover the minimax loss function, the iterative training procedure, common challenges like mode collapse and training instability, and briefly mention popular GAN variants (e.g., WGAN, StyleGAN).",
        "Write a well-structured analytical essay discussing the multifaceted impact of social media platforms (like Facebook, Twitter, Instagram, TikTok) on modern society. Explore effects on communication patterns, interpersonal relationships, political discourse and activism, mental health and well-being, the spread of information and misinformation, consumer behavior, and cultural trends. Present arguments supported by examples or evidence.",
        "Analyze the key conceptual differences and practical implications separating the three main paradigms of machine learning: Supervised Learning (learning from labeled data for classification/regression), Unsupervised Learning (finding patterns in unlabeled data like clustering/dimensionality reduction), and Reinforcement Learning (learning through trial-and-error via rewards/penalties in an environment). Provide clear examples for each paradigm.",
        "Provide a detailed, step-by-step practical guide for a non-expert user on how to set up and maintain a secure home Wi-Fi network. Cover aspects like choosing a strong router password, changing the default network name (SSID), enabling WPA2/WPA3 encryption, setting up a guest network, keeping router firmware updated, disabling WPS, and considering MAC address filtering or VPN usage.",
        "Discuss the profound historical significance and the wide-ranging, long-term consequences of the invention of the printing press by Johannes Gutenberg in the 15th century. Analyze its impact on literacy rates, the dissemination of knowledge and ideas (including the Reformation and Scientific Revolution), the standardization of languages, political power structures, and the foundations of the modern information age.",
        "Summarize the principal arguments frequently presented both in favor of and against continued publicly funded space exploration initiatives. Consider scientific discovery benefits, technological advancements (spin-offs), national prestige, potential for resource acquisition or colonization, inspiration for STEM education, versus the high costs, inherent risks, ethical considerations, and arguments for prioritizing terrestrial problems.",
        "Write an introductory tutorial demonstrating the fundamental usage of the Pandas library in Python for basic data analysis tasks. Cover creating DataFrames and Series, reading data from files (e.g., CSV), selecting and filtering data (using loc, iloc, boolean indexing), handling missing values, performing basic descriptive statistics (.describe(), .mean(), .value_counts()), and grouping data (.groupby()).",
        "Explain the core concept of differential privacy as a formal mathematical definition of privacy protection in data analysis. Describe the roles of the privacy budget (epsilon) and sensitivity. Illustrate how noise (e.g., Laplacian or Gaussian mechanism) is added to query results or model parameters (like in DP-SGD) to achieve privacy guarantees while balancing data utility. Mention common applications.",
        "Generate a detailed and well-structured project proposal outlining the development plan for a hypothetical mobile application designed to help users track their daily water intake and promote better hydration habits. Include sections on Introduction/Problem Statement, Proposed Solution/Features, Target Audience, Technical Approach (platform, key technologies), Project Timeline/Milestones, Resource Requirements, and Evaluation Metrics.",
        "Analyze the fundamental economic principles that underpin international trade between nations. Discuss concepts like comparative advantage (Ricardo), absolute advantage (Smith), specialization, economies of scale, trade barriers (tariffs, quotas), trade agreements, balance of trade, exchange rates, and the overall benefits and drawbacks of globalization from an economic perspective.",
        "Discuss the critical role of ethics in the field of artificial intelligence research and development. Explore key ethical considerations such as algorithmic bias and fairness, transparency and explainability (XAI), accountability for AI decisions, data privacy concerns, potential job displacement, the safe development of Artificial General Intelligence (AGI), and the importance of establishing ethical guidelines and regulations.",
        "Write a comprehensive review summarizing a significant recent scientific breakthrough or discovery reported in a reputable journal within the last year (e.g., in fields like medicine, astrophysics, materials science, biotechnology). Explain the background, the methods used, the key findings, the potential implications or applications, and any remaining questions or future research directions."
    ]
    dataset_c4 = [text for text in long_job_c_base] * (30 // len(long_job_c_base))
     # Ensure correct length if division is not exact
    if len(dataset_c4) < 30:
        dataset_c4.extend([long_job_c_base[i % len(long_job_c_base)] for i in range(len(dataset_c4), 30)])


    scenario = [
        # --- Job C1 (Very Short) ---
        ScenarioEvent(delay_seconds=0.1, event_type='TRAIN_JOB', details={
            "job_id": "scenC_train_1_vshort", "base_model_id": base_model_name,
            "dataset_samples": dataset_c1, "max_steps": 5,
            "training_params": {"lr": 5e-5, "batch_size": 1},
            "adapter_save_path": "./adapters/scenC_train_1_vshort"
        }),
        # --- Job C2 (Medium) ---
        ScenarioEvent(delay_seconds=0.5, event_type='TRAIN_JOB', details={
            "job_id": "scenC_train_2_medium", "base_model_id": base_model_name,
            "dataset_samples": dataset_c2, "max_steps": 15,
            "training_params": {"lr": 4e-5, "batch_size": 1},
            "adapter_save_path": "./adapters/scenC_train_2_medium"
        }),
        # --- Inference Stream 1 ---
        ScenarioEvent(delay_seconds=1.0, event_type='INFERENCE_REQ', details={
            "prompt": "Explain the concept of quantum entanglement.", "adapter_path": None
        }),
        ScenarioEvent(delay_seconds=2.5, event_type='INFERENCE_REQ', details={
            "prompt": "What is the currency of Japan?", "adapter_path": None
        }),
        # --- Job C3 (Short) ---
        ScenarioEvent(delay_seconds=3.0, event_type='TRAIN_JOB', details={
            "job_id": "scenC_train_3_short", "base_model_id": base_model_name,
            "dataset_samples": dataset_c3, "max_steps": 8,
            "training_params": {"lr": 6e-5, "batch_size": 1},
            "adapter_save_path": "./adapters/scenC_train_3_short"
        }),
        ScenarioEvent(delay_seconds=4.0, event_type='INFERENCE_REQ', details={
            "prompt": "List three benefits of using Python.", "adapter_path": None # Base model inference
        }),
        # --- Inference Stream 2 ---
         ScenarioEvent(delay_seconds=6.0, event_type='INFERENCE_REQ', details={
            "prompt": "Who painted the Mona Lisa?", "adapter_path": None
        }),
        ScenarioEvent(delay_seconds=8.0, event_type='INFERENCE_REQ', details={
            "prompt": "Translate 'hello world' to French.", "adapter_path": None
        }),
         # --- Job C4 (Long) ---
        ScenarioEvent(delay_seconds=10.0, event_type='TRAIN_JOB', details={
            "job_id": "scenC_train_4_long", "base_model_id": base_model_name,
            "dataset_samples": dataset_c4, "max_steps": 30,
            "training_params": {"lr": 3e-5, "batch_size": 1},
            "adapter_save_path": "./adapters/scenC_train_4_long"
        }),
        # --- Inference Stream 3 (During Long Job) ---
        ScenarioEvent(delay_seconds=12.0, event_type='INFERENCE_REQ', details={
            "prompt": "What year did World War II end?", "adapter_path": None
        }),
        ScenarioEvent(delay_seconds=15.0, event_type='INFERENCE_REQ', details={
            "prompt": "Summarize the first chapter of '1984'.", "adapter_path": None
        }),
         ScenarioEvent(delay_seconds=18.0, event_type='INFERENCE_REQ', details={
            "prompt": "What is the chemical symbol for Gold?", "adapter_path": None
        }),
        ScenarioEvent(delay_seconds=22.0, event_type='INFERENCE_REQ', details={
            "prompt": "Describe the process of photosynthesis.", "adapter_path": None
        }),
    ]
    return scenario
# --- End New Scenario Definition ---


if __name__ == "__main__":
    # 1. Setup Logger
    # If logger config is in lora-co-serving/configs/logging_config.yaml
    # log_config_path = os.path.join(os.path.dirname(__file__), '..', 'configs', 'logging_config.yaml') 
    # setup_logger(config_path=log_config_path, log_level=logging.INFO) 
    setup_logger(log_level=logging.INFO) # Use INFO level for eval clarity
    logger = logging.getLogger("EvaluationRun")
    logger.info("--- Starting Evaluation Run --- ")

    # 2. Load Configuration
    # Assuming config is in lora-co-serving/configs/config.yaml and load_config handles default path
    config: SystemConfig = load_config()
    if not config:
        logger.error("Failed to load configuration. Exiting.")
        sys.exit(1)
    logger.info(f"Using configuration: {config}")

    # 3. Load Base Model
    logger.info(f"Loading base model: {config.model.name}")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model, tokenizer = load_base_model(config.model.name, device=device)
    if model is None or tokenizer is None: sys.exit(1)
    logger.info("Base model and tokenizer loaded.")

    # 4. Prepare Base Model for LoRA Training
    logger.info("Setting up base model for LoRA training...")
    try:
        lora_config_dict = asdict(config.training.lora_config)
        model = setup_lora_training(model, tokenizer, lora_config_dict)
        if model is None: raise ValueError("setup_lora_training returned None")
        logger.info("Base model prepared for LoRA training.")
    except Exception as e:
        logger.error(f"Failed to setup base model for LoRA training: {e}", exc_info=True)
        sys.exit(1)

    # 5. Initialize Managers and Engine
    logger.info("Initializing managers and inference engine...")
    queue_manager = QueueManager(config.queue.max_inference_queue_size, config.queue.max_training_queue_size)
    adapter_manager = LoRAAdapterManager(model, config.managers.adapter_cache_size)
    job_manager = ActiveTrainingJobManager()
    engine = InferenceEngine(model, tokenizer, adapter_manager, device)
    logger.info("Managers and Inference Engine initialized.")

    # 6. Initialize Prioritization Strategy based on Config
    strategy_name = config.prioritization.strategy
    strategy_params = config.prioritization.params if hasattr(config.prioritization, 'params') and config.prioritization.params else {}
    logger.info(f"Initializing '{strategy_name}' prioritization strategy...")
    if strategy_name == "RoundRobin":
        prioritization_strategy = RoundRobinStrategy(**strategy_params)
    elif strategy_name == "ForwardStagnationAware":
        prioritization_strategy = ForwardStagnationAwareStrategy(**strategy_params)
    elif strategy_name == "ReverseStagnationAware":
        prioritization_strategy = ReverseStagnationAwareStrategy(**strategy_params)
    elif strategy_name == "LeastProgressFirst":
        prioritization_strategy = LeastProgressFirstStrategy(**strategy_params)
    else:
        logger.warning(f"Unknown prioritization strategy '{strategy_name}'. Using RoundRobin fallback.")
        prioritization_strategy = RoundRobinStrategy()
    logger.info(f"Using strategy: {type(prioritization_strategy).__name__} with params: {strategy_params}")

    # 7. Initialize Main Controller
    logger.info("Initializing Main Controller...")
    controller = MainController(
        config=config,
        engine=engine,
        job_manager=job_manager,
        adapter_manager=adapter_manager,
        queue_manager=queue_manager,
        prioritization_strategy=prioritization_strategy,
        device=device
    )
    logger.info("Main Controller initialized.")

    # 8. Define and Start Scenario
    scenario = define_scenario_c(config.model.name)
    scenario_job_ids = [event.details['job_id'] for event in scenario if event.event_type == 'TRAIN_JOB']
    logger.info(f"Defined Scenario C with training jobs: {scenario_job_ids}")
    
    submitter_thread = threading.Thread(target=scenario_submitter, args=(queue_manager, scenario), daemon=True)
    submitter_thread.start()
    logger.info("Scenario submitter thread started.")

    # 9. Start Controller Loop
    logger.info("Starting controller loop in background thread...")
    controller_thread = threading.Thread(target=controller.run_loop, daemon=True)
    controller_thread.start()
    logger.info("Controller thread started.")

    # 10. Monitor for Scenario Completion (All Training Jobs Finished)
    logger.info("Monitoring training jobs for completion...")
    start_monitor_time = time.time()
    all_jobs_finished = False
    scenario_start_time = start_monitor_time # Approximate start
    
    try:
        while not all_jobs_finished and controller_thread.is_alive():
            time.sleep(2) # Check every 2 seconds
            # Check job statuses
            all_jobs_finished = True # Assume finished until proven otherwise
            all_jobs_seen = True 
            for job_id in scenario_job_ids:
                state = job_manager.get_job_state(job_id)
                if not state:
                    all_jobs_seen = False # A job hasn't even registered yet
                    all_jobs_finished = False
                    break # No need to check others if one is missing
                elif state.status not in [JobStatus.COMPLETED, JobStatus.FAILED]:
                    all_jobs_finished = False
                    # Don't break here, allow logging completion of others
            
            # Log completions as they happen (moved check inside loop)
            current_job_states = job_manager.get_all_job_states()
            completed_ids = {jid for jid, s in current_job_states.items() if s.status == JobStatus.COMPLETED and jid in scenario_job_ids}
            failed_ids = {jid for jid, s in current_job_states.items() if s.status == JobStatus.FAILED and jid in scenario_job_ids}
            # logger.info(f"Monitor check: Completed={completed_ids}, Failed={failed_ids}") # Debug
            
            # Ensure all jobs defined in the scenario have appeared in the job manager at least once
            if not all_jobs_seen:
                 all_jobs_finished = False
            
    except KeyboardInterrupt:
        logger.info("KeyboardInterrupt received during monitoring.")
    finally:
        # 11. Stop Controller and Collect Results
        logger.info("Scenario monitoring finished or interrupted.")
        run_end_time = time.time()
        if controller.running:
            logger.info("Requesting controller stop...")
            controller.stop_loop()
            logger.info("Waiting for controller thread to finish...")
            controller_thread.join(timeout=10.0)
            if controller_thread.is_alive():
                 logger.warning("Controller thread did not stop gracefully.")
        
        logger.info("--- Evaluation Run Finished --- ")
        total_monitor_time = run_end_time - start_monitor_time

        # --- Metric Collection --- 
        logger.info("Collecting final metrics...")
        
        # Get Job Timings from JobManager
        final_job_states = job_manager.get_all_job_states()
        training_job_results = {}
        job_start_times = []
        job_end_times = []
        job_completion_durations = []

        for job_id in scenario_job_ids:
            state = final_job_states.get(job_id)
            if state:
                # Calculate duration manually
                duration = None
                if state.end_time and state.start_time:
                    duration = state.end_time - state.start_time
                    
                training_job_results[job_id] = {
                    "status": state.status.name,
                    "start_time": state.start_time,
                    "end_time": state.end_time,
                    "duration_seconds": duration, # Use calculated duration
                    "final_step": state.current_step,
                    "error": state.error_message
                }
                if state.start_time: job_start_times.append(state.start_time)
                if state.end_time: job_end_times.append(state.end_time)
                # Use calculated duration for summary stats
                if duration is not None: job_completion_durations.append(duration)
            else:
                 training_job_results[job_id] = {"status": "NOT_FOUND"}

        # Get Inference Results from Controller
        inference_results_raw = controller.get_completed_inference_results()
        inference_metrics = {
            "total_requests_processed": len(inference_results_raw),
            "successful_requests": 0,
            "failed_requests": 0,
            "end_to_end_latencies_ms": [],
            "processing_latencies_ms": []
        }
        for res in inference_results_raw:
            if res.get('error') is None and res.get('output') is not None:
                inference_metrics["successful_requests"] += 1
                if res.get('processing_end_time') and res.get('submission_time'):
                    e2e_latency = (res['processing_end_time'] - res['submission_time']) * 1000 # ms
                    inference_metrics["end_to_end_latencies_ms"].append(e2e_latency)
                if res.get('processing_end_time') and res.get('processing_start_time'):
                    proc_latency = (res['processing_end_time'] - res['processing_start_time']) * 1000 # ms
                    inference_metrics["processing_latencies_ms"].append(proc_latency)
            else:
                inference_metrics["failed_requests"] += 1

        # --- Calculate Summary Metrics --- 
        summary_metrics = {}
        
        # Training Metrics
        summary_metrics["training_total_jobs_in_scenario"] = len(scenario_job_ids)
        summary_metrics["training_completed_jobs"] = len([r for r in training_job_results.values() if r["status"] == "COMPLETED"])
        summary_metrics["training_failed_jobs"] = len([r for r in training_job_results.values() if r["status"] == "FAILED"])
        
        if job_start_times and job_end_times:
            first_job_start = min(job_start_times)
            last_job_end = max(job_end_times)
            summary_metrics["training_makespan_seconds"] = last_job_end - first_job_start
        else:
             summary_metrics["training_makespan_seconds"] = None
             
        if job_completion_durations:
             summary_metrics["training_avg_completion_time_seconds"] = np.mean(job_completion_durations)
             summary_metrics["training_stddev_completion_time_seconds"] = np.std(job_completion_durations)
        else:
             summary_metrics["training_avg_completion_time_seconds"] = None
             summary_metrics["training_stddev_completion_time_seconds"] = None

        # Inference Metrics
        summary_metrics["inference_total_processed"] = inference_metrics["total_requests_processed"]
        summary_metrics["inference_successful"] = inference_metrics["successful_requests"]
        summary_metrics["inference_failed"] = inference_metrics["failed_requests"]
        total_run_duration_seconds = run_end_time - scenario_start_time
        if total_run_duration_seconds > 0:
             summary_metrics["inference_throughput_success_per_sec"] = inference_metrics["successful_requests"] / total_run_duration_seconds
        else:
             summary_metrics["inference_throughput_success_per_sec"] = 0

        if inference_metrics["end_to_end_latencies_ms"]:
            e2e = inference_metrics["end_to_end_latencies_ms"]
            summary_metrics["inference_e2e_latency_ms_avg"] = np.mean(e2e)
            summary_metrics["inference_e2e_latency_ms_p50"] = np.percentile(e2e, 50)
            summary_metrics["inference_e2e_latency_ms_p95"] = np.percentile(e2e, 95)
            summary_metrics["inference_e2e_latency_ms_p99"] = np.percentile(e2e, 99)
        else: 
             summary_metrics["inference_e2e_latency_ms_avg"] = None
             # ... set other percentiles to None ...

        if inference_metrics["processing_latencies_ms"]:
            proc = inference_metrics["processing_latencies_ms"]
            summary_metrics["inference_processing_latency_ms_avg"] = np.mean(proc)
            summary_metrics["inference_processing_latency_ms_p50"] = np.percentile(proc, 50)
            summary_metrics["inference_processing_latency_ms_p95"] = np.percentile(proc, 95)
            summary_metrics["inference_processing_latency_ms_p99"] = np.percentile(proc, 99)
        else:
             summary_metrics["inference_processing_latency_ms_avg"] = None
             # ... set other percentiles to None ...

        # --- Print Summary --- 
        logger.info("--- Evaluation Summary Metrics ---")
        logger.info(f"Strategy Used: {strategy_name} (Params: {strategy_params})")
        logger.info(f"Total Monitoring Time: {total_monitor_time:.2f} seconds")
        logger.info("[Training]")
        logger.info(f"  Jobs in Scenario: {summary_metrics['training_total_jobs_in_scenario']}")
        logger.info(f"  Completed: {summary_metrics['training_completed_jobs']}")
        logger.info(f"  Failed: {summary_metrics['training_failed_jobs']}")
        logger.info(f"  Makespan (s): {summary_metrics['training_makespan_seconds']:.2f}" if summary_metrics['training_makespan_seconds'] is not None else "  Makespan (s): N/A")
        logger.info(f"  Avg Completion Time (s): {summary_metrics['training_avg_completion_time_seconds']:.2f}" if summary_metrics['training_avg_completion_time_seconds'] is not None else "  Avg Completion Time (s): N/A")
        logger.info(f"  Std Dev Completion Time (s): {summary_metrics['training_stddev_completion_time_seconds']:.2f}" if summary_metrics['training_stddev_completion_time_seconds'] is not None else "  Std Dev Completion Time (s): N/A")
        logger.info("[Inference]")
        logger.info(f"  Total Processed: {summary_metrics['inference_total_processed']}")
        logger.info(f"  Successful: {summary_metrics['inference_successful']}")
        logger.info(f"  Failed: {summary_metrics['inference_failed']}")
        logger.info(f"  Throughput (success/sec): {summary_metrics['inference_throughput_success_per_sec']:.2f}")
        logger.info(f"  E2E Latency Avg (ms): {summary_metrics['inference_e2e_latency_ms_avg']:.2f}" if summary_metrics['inference_e2e_latency_ms_avg'] else "  E2E Latency Avg (ms): N/A")
        logger.info(f"  E2E Latency P50 (ms): {summary_metrics['inference_e2e_latency_ms_p50']:.2f}" if summary_metrics['inference_e2e_latency_ms_p50'] else "  E2E Latency P50 (ms): N/A")
        logger.info(f"  E2E Latency P95 (ms): {summary_metrics['inference_e2e_latency_ms_p95']:.2f}" if summary_metrics['inference_e2e_latency_ms_p95'] else "  E2E Latency P95 (ms): N/A")
        logger.info(f"  E2E Latency P99 (ms): {summary_metrics['inference_e2e_latency_ms_p99']:.2f}" if summary_metrics['inference_e2e_latency_ms_p99'] else "  E2E Latency P99 (ms): N/A")
        logger.info(f"  Processing Latency Avg (ms): {summary_metrics['inference_processing_latency_ms_avg']:.2f}" if summary_metrics['inference_processing_latency_ms_avg'] else "  Processing Latency Avg (ms): N/A")
        # ... Add P50/P95/P99 for processing latency if desired ...

        # --- Save Detailed Results --- 
        results_data = {
            "run_timestamp": datetime.now().isoformat(),
            "strategy": strategy_name,
            "strategy_params": strategy_params,
            "config_used": asdict(config),
            "total_monitor_time_seconds": total_monitor_time,
            "summary_metrics": summary_metrics,
            "training_job_details": training_job_results,
            "inference_request_details": inference_results_raw # Contains raw timings and outputs
        }
        
        results_dir = "evaluation_results"
        os.makedirs(results_dir, exist_ok=True)
        results_filename = f"eval_{strategy_name}_{datetime.now():%Y%m%d_%H%M%S}.json"
        results_path = os.path.join(results_dir, results_filename)
        
        try:
            with open(results_path, 'w') as f:
                json.dump(results_data, f, indent=4, default=str) # Use default=str for non-serializable like Enum?
            logger.info(f"Detailed evaluation results saved to: {results_path}")
        except Exception as e:
            logger.error(f"Failed to save results to JSON file {results_path}: {e}", exc_info=True)

        sys.exit(0) # Exit cleanly 