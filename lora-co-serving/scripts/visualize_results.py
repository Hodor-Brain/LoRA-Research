# visualize_results.py

import json
import os
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.ticker as mticker

# --- Configuration ---
# List of result file paths relative to the script location or project root
# Adjust these paths if your results are elsewhere or filenames differ
RESULT_FILES = {
    "RoundRobin": "../evaluation_results/eval_RoundRobin_20250430_230923.json",
    "LeastProgressFirst": "../evaluation_results/eval_LeastProgressFirst_20250430_230649.json",
    "ForwardStagnationAware": "../evaluation_results/eval_ForwardStagnationAware_20250430_230813.json",
}

# Scenario C Job IDs
SCENARIO_C_JOBS = [
    'scenC_train_1_vshort', 
    'scenC_train_2_medium', 
    'scenC_train_3_short', 
    'scenC_train_4_long'
]

OUTPUT_DIR = "../evaluation_results" # Where to save the plots
METRICS_PLOT_FILENAME = "scenario_c_metrics_comparison_ukr.png"
TIMELINE_PLOT_FILENAME = "scenario_c_job_timeline_ukr.png"

# Ukrainian Translations for Plot Elements
UKR_TRANSLATIONS = {
    "Makespan (s)": 'Час виконання (с)',
    "Avg Job Comp (s)": 'Сер. Заверш. Завдання (с)',
    "Avg Inf Latency (ms)": 'Сер. E2E Затримка Інф. (мс)',
    "Inf Throughput (req/s)": 'Проп. Здатність Інф. (зап/с)',
    "Scenario C: Performance Metrics Comparison": 'Порівняння Метрик Продуктивності',
    "Scenario C: Training Job Execution Timeline": 'Часова Шкала Виконання Навчальних Завдань',
    "Scenario C: Job Duration Comparison by Strategy": 'Порівняння Тривалості Завдань за Стратегіями',
    "Scenario C: Distribution of End-to-End Inference Latency by Strategy": 'Розподіл End-to-End Затримки Інференсу за Стратегіями',
    "Scenario C: Cumulative Successful Inference Requests Over Time": 'Кумулятивна Кількість Успішних Запитів на Інференс з Часом',
    "Time (seconds since first job start)": "Час (секунди від старту першого завдання)",
    "Duration (seconds)": 'Тривалість (секунди)',
    "E2E Latency (ms)": 'E2E Затримка (мс)',
    "Time (seconds relative to first job start)": 'Час (секунди відносно старту першого завдання)',
    "Cumulative Successful Inference Requests": 'Кумулятивна К-ть Успішних Запитів Інференсу',
    "Job ID": "ID Завдання",
    "Strategy": "Стратегія",
    "(No data)": "(Немає даних)",
    " by Strategy": " за Стратегіями"
}

# --- Data Loading ---
def load_results(filepath):
    """Loads evaluation results from a JSON file."""
    try:
        # Construct path relative to this script's directory
        script_dir = os.path.dirname(__file__)
        full_path = os.path.abspath(os.path.join(script_dir, filepath))
        with open(full_path, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Error: Result file not found at {full_path}")
        return None
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from {full_path}")
        return None

all_results_data = {}
for strategy, filepath in RESULT_FILES.items():
    data = load_results(filepath)
    if data:
        all_results_data[strategy] = data

if not all_results_data:
    print("No valid result files loaded. Exiting.")
    exit()

# --- Data Extraction ---
strategies = list(all_results_data.keys())
metrics_data = {
    "Makespan (s)": [],
    "Avg Job Comp (s)": [],
    "Avg Inf Latency (ms)": [],
    "Inf Throughput (req/s)": []
}
job_timelines = {} # {strategy: {job_id: (start, duration)}}

min_start_time = float('inf')

for strategy in strategies:
    summary = all_results_data[strategy]["summary_metrics"]
    details = all_results_data[strategy]["training_job_details"]
    
    metrics_data["Makespan (s)"].append(summary.get("training_makespan_seconds", 0))
    metrics_data["Avg Job Comp (s)"].append(summary.get("training_avg_completion_time_seconds", 0))
    metrics_data["Avg Inf Latency (ms)"].append(summary.get("inference_e2e_latency_ms_avg", 0))
    metrics_data["Inf Throughput (req/s)"].append(summary.get("inference_throughput_success_per_sec", 0))

    job_times = {}
    for job_id in SCENARIO_C_JOBS:
        job_detail = details.get(job_id)
        if job_detail and job_detail.get("start_time") and job_detail.get("duration_seconds"):
            start = job_detail["start_time"]
            duration = job_detail["duration_seconds"]
            job_times[job_id] = (start, duration)
            min_start_time = min(min_start_time, start) # Find earliest start time across all jobs/strategies
        else:
            print(f"Warning: Missing start_time or duration for {job_id} in {strategy}")
            job_times[job_id] = (0, 0) # Placeholder if data missing
            
    job_timelines[strategy] = job_times

# Adjust start times to be relative to the earliest start time
for strategy in strategies:
    for job_id in SCENARIO_C_JOBS:
        start, duration = job_timelines[strategy][job_id]
        if start > 0: # Adjust only if valid start time exists
           job_timelines[strategy][job_id] = (start - min_start_time, duration)

# --- New Data Extraction for Inference Plots ---
all_inference_latencies = {} # {strategy: [latency1, latency2, ...]}
all_inference_completion_times = {} # {strategy: [rel_comp_time1, rel_comp_time2, ...]}

for strategy in strategies:
    latencies = []
    completion_times = []
    inference_details = all_results_data[strategy].get("inference_request_details", [])
    for req in inference_details:
        # Check for successful completion and valid timing data
        if req.get('error') is None and req.get('output') is not None and \
           req.get('processing_end_time') is not None and req.get('submission_time') is not None:
            latency_ms = (req['processing_end_time'] - req['submission_time']) * 1000
            latencies.append(latency_ms)
            # Calculate completion time relative to the earliest job start
            completion_time_abs = req['processing_end_time']
            completion_time_rel = completion_time_abs - min_start_time 
            completion_times.append(completion_time_rel)
            
    all_inference_latencies[strategy] = latencies
    all_inference_completion_times[strategy] = sorted(completion_times) # Sort for cumulative plot

# --- Plotting ---

# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)
plot_metrics_path = os.path.join(OUTPUT_DIR, METRICS_PLOT_FILENAME)
plot_timeline_path = os.path.join(OUTPUT_DIR, TIMELINE_PLOT_FILENAME)


# 1. Summary Metrics Bar Chart
print("Generating Summary Metrics Plot (Ukrainian)...")
labels = strategies
metrics_to_plot = list(metrics_data.keys())
x = np.arange(len(labels))  # the label locations
n_metrics = len(metrics_to_plot)
width = 0.8 / n_metrics # width of the bars

fig, axes = plt.subplots(2, 2, figsize=(12, 10)) # Create a 2x2 grid of subplots
axes = axes.flatten() # Flatten the 2x2 array for easy iteration

for i, metric_name_en in enumerate(metrics_to_plot):
    ax = axes[i]
    values = metrics_data[metric_name_en]
    metric_name_ukr = UKR_TRANSLATIONS.get(metric_name_en, metric_name_en) # Translate
    rects = ax.bar(x, values, width * n_metrics, label=metric_name_ukr, color=plt.cm.Paired(np.linspace(0, 1, len(labels)))) # Different color per strategy
    
    ax.set_ylabel(metric_name_ukr) # Use translated label
    ax.set_title(f'{metric_name_ukr}{UKR_TRANSLATIONS[" by Strategy"]}') # Use translated title parts
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=10, ha='right')
    # ax.legend() # Remove individual legends, use a single figure legend

    # Add value labels on top of bars
    ax.bar_label(rects, padding=3, fmt='%.2f')

fig.suptitle(UKR_TRANSLATIONS["Scenario C: Performance Metrics Comparison"], fontsize=16, y=1.02) # Use translated title
fig.tight_layout(pad=3.0)
# fig.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=n_metrics) # Add a single legend below
plt.savefig(plot_metrics_path, bbox_inches='tight')
print(f"Metrics plot saved to: {plot_metrics_path}")
plt.close(fig)


# 2. Job Execution Timeline (Gantt-like)
print("Generating Job Timeline Plot (Ukrainian)...")
fig, ax = plt.subplots(figsize=(12, 8))

y_ticks = []
y_labels = []
bar_height = 0.6
job_colors = plt.cm.viridis(np.linspace(0, 1, len(SCENARIO_C_JOBS))) # Color by job

current_y = 0
y_pos_map = {} # strategy -> y_base

for i, strategy in enumerate(strategies):
     y_pos_map[strategy] = current_y
     # Add strategy label
     y_ticks.append(current_y + (len(SCENARIO_C_JOBS) * bar_height) / 2 - bar_height/2) # Center label
     y_labels.append(strategy) # Keep strategy name in English
     current_y += (len(SCENARIO_C_JOBS) + 1) * bar_height # Add spacing between strategies

for i, strategy in enumerate(strategies):
     y_base = y_pos_map[strategy]
     for j, job_id in enumerate(SCENARIO_C_JOBS):
         start, duration = job_timelines[strategy][job_id]
         y_pos = y_base + j * bar_height
         color = job_colors[j]
         ax.barh(y_pos, duration, left=start, height=bar_height, align='center', color=color, edgecolor='black', label=job_id if i == 0 else "") # Only label once per job
         # Add job ID text inside the bar if it fits
         if duration > 0: # Avoid division by zero or tiny bars
             text_x = start + duration / 2
             ax.text(text_x, y_pos, job_id.replace("scenC_train_", ""), va='center', ha='center', color='white' if duration > 5 else 'black', fontsize=8)


ax.set_yticks(y_ticks)
ax.set_yticklabels(y_labels)
ax.invert_yaxis() # Strategies top-to-bottom
ax.set_xlabel(UKR_TRANSLATIONS["Time (seconds since first job start)"]) # Use translated label
ax.set_title(UKR_TRANSLATIONS["Scenario C: Training Job Execution Timeline"]) # Use translated title

# Improve x-axis readability
ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True, nbins=10))
ax.grid(axis='x', linestyle='--', alpha=0.6)

# Create a custom legend for job colors outside the loop to avoid duplicates
handles = [plt.Rectangle((0,0),1,1, color=job_colors[j]) for j in range(len(SCENARIO_C_JOBS))]
ax.legend(handles, SCENARIO_C_JOBS, title=UKR_TRANSLATIONS["Job ID"], bbox_to_anchor=(1.02, 1), loc='upper left') # Use translated legend title

plt.tight_layout(rect=[0, 0, 0.85, 1]) # Adjust layout to make space for legend
plt.savefig(plot_timeline_path, bbox_inches='tight')
print(f"Timeline plot saved to: {plot_timeline_path}")
plt.close(fig)


# 3. Grouped Bar Chart: Job Durations
print("Generating Job Durations Plot (Ukrainian)...")
plot_durations_path = os.path.join(OUTPUT_DIR, "scenario_c_job_durations_ukr.png") # Added _ukr
job_ids_short = [j.replace("scenC_train_", "") for j in SCENARIO_C_JOBS]
x_jobs = np.arange(len(job_ids_short)) # the label locations for jobs
n_strategies = len(strategies)
width_jobs = 0.8 / n_strategies # width of the bars

fig_dur, ax_dur = plt.subplots(figsize=(12, 7))
strategy_colors = plt.cm.Paired(np.linspace(0, 1, n_strategies))

for i, strategy in enumerate(strategies):
    durations = [job_timelines[strategy][job_id][1] for job_id in SCENARIO_C_JOBS] # Get duration (index 1)
    offset = width_jobs * (i - n_strategies / 2 + 0.5)
    rects = ax_dur.bar(x_jobs + offset, durations, width_jobs, label=strategy, color=strategy_colors[i]) # Keep strategy name in English
    ax_dur.bar_label(rects, padding=3, fmt='%.1f', fontsize=8)

# Add labels, title and axes ticks
ax_dur.set_ylabel(UKR_TRANSLATIONS['Duration (seconds)']) # Use translated label
ax_dur.set_title(UKR_TRANSLATIONS['Scenario C: Job Duration Comparison by Strategy']) # Use translated title
ax_dur.set_xticks(x_jobs)
ax_dur.set_xticklabels(job_ids_short, rotation=15, ha='right')
ax_dur.legend(title=UKR_TRANSLATIONS["Strategy"], bbox_to_anchor=(1.02, 1), loc='upper left') # Use translated legend title
ax_dur.grid(axis='y', linestyle='--', alpha=0.6)

plt.tight_layout(rect=[0, 0, 0.85, 1]) # Adjust layout for legend
plt.savefig(plot_durations_path, bbox_inches='tight')
print(f"Job durations plot saved to: {plot_durations_path}")
plt.close(fig_dur)


# 4. Grouped Bar Chart: Job Start Times (Relative)
# print("Generating Job Start Times Plot...")
# plot_starts_path = os.path.join(OUTPUT_DIR, "scenario_c_job_start_times.png")
# # Reuse job_ids_short, x_jobs, n_strategies, width_jobs, strategy_colors from previous plot
# 
# fig_start, ax_start = plt.subplots(figsize=(12, 7))
# 
# for i, strategy in enumerate(strategies):
#     start_times = [job_timelines[strategy][job_id][0] for job_id in SCENARIO_C_JOBS] # Get start time (index 0)
#     offset = width_jobs * (i - n_strategies / 2 + 0.5)
#     rects = ax_start.bar(x_jobs + offset, start_times, width_jobs, label=strategy, color=strategy_colors[i])
#     ax_start.bar_label(rects, padding=3, fmt='%.1f', fontsize=8)
# 
# # Add labels, title and axes ticks
# ax_start.set_ylabel('Start Time (seconds relative to first job)')
# ax_start.set_title('Scenario C: Job Relative Start Time Comparison by Strategy')
# ax_start.set_xticks(x_jobs)
# ax_start.set_xticklabels(job_ids_short, rotation=15, ha='right')
# ax_start.legend(title="Strategy", bbox_to_anchor=(1.02, 1), loc='upper left')
# ax_start.grid(axis='y', linestyle='--', alpha=0.6)
# 
# plt.tight_layout(rect=[0, 0, 0.85, 1]) # Adjust layout for legend
# plt.savefig(plot_starts_path, bbox_inches='tight')
# print(f"Job start times plot saved to: {plot_starts_path}")
# plt.close(fig_start)


# 5. Inference Latency Distribution (Box Plot)
print("Generating Inference Latency Distribution Plot (Ukrainian)...")
plot_latency_box_path = os.path.join(OUTPUT_DIR, "scenario_c_inf_latency_distribution_ukr.png") # Added _ukr

latency_data_to_plot = [all_inference_latencies.get(s, []) for s in strategies]

fig_box, ax_box = plt.subplots(figsize=(10, 6))

# Check if there is data to plot
if any(latency_data_to_plot):
    box = ax_box.boxplot(latency_data_to_plot, labels=strategies, patch_artist=True, showfliers=True) # Keep strategy name in English
    
    # Set colors for the boxes
    colors = plt.cm.Paired(np.linspace(0, 1, len(strategies)))
    for patch, color in zip(box['boxes'], colors):
        patch.set_facecolor(color)
        
    ax_box.set_title(UKR_TRANSLATIONS['Scenario C: Distribution of End-to-End Inference Latency by Strategy']) # Use translated title
    ax_box.set_ylabel(UKR_TRANSLATIONS['E2E Latency (ms)']) # Use translated label
    ax_box.grid(axis='y', linestyle='--', alpha=0.7)
else:
    ax_box.text(0.5, 0.5, 'Не знайдено дійсних даних про затримку інференсу.', # Translated message
                horizontalalignment='center', verticalalignment='center', 
                transform=ax_box.transAxes)
    ax_box.set_title(UKR_TRANSLATIONS['Scenario C: Distribution of End-to-End Inference Latency by Strategy']) # Use translated title
    ax_box.set_ylabel(UKR_TRANSLATIONS['E2E Latency (ms)']) # Use translated label

plt.tight_layout()
plt.savefig(plot_latency_box_path, bbox_inches='tight')
print(f"Inference latency distribution plot saved to: {plot_latency_box_path}")
plt.close(fig_box)


# 6. Cumulative Successful Inference Requests Over Time
print("Generating Cumulative Inference Requests Plot (Ukrainian)...")
plot_cumulative_inf_path = os.path.join(OUTPUT_DIR, "scenario_c_cumulative_inferences_ukr.png") # Added _ukr

fig_cum, ax_cum = plt.subplots(figsize=(10, 6))

max_time = 0 # Find the max time across all strategies for axis limit

strategy_colors_line = plt.cm.Paired(np.linspace(0, 1, len(strategies)))

for i, strategy in enumerate(strategies):
    completion_times = all_inference_completion_times.get(strategy, [])
    if completion_times:
        # Create cumulative counts
        counts = np.arange(1, len(completion_times) + 1)
        # Add a point at time 0 with count 0 for a proper step plot start
        plot_times = np.concatenate(([0], completion_times))
        plot_counts = np.concatenate(([0], counts))
        ax_cum.plot(plot_times, plot_counts, drawstyle='steps-post', label=strategy, color=strategy_colors_line[i], linewidth=2) # Keep strategy name in English
        max_time = max(max_time, completion_times[-1] if completion_times else 0)
    else:
         no_data_label = f'{strategy} {UKR_TRANSLATIONS["(No data)"]}'
         ax_cum.plot([], [], label=no_data_label, color=strategy_colors_line[i]) # Add translated label even if no data

if max_time > 0:
    ax_cum.set_xlim(0, max_time * 1.05) # Add a bit of padding
    ax_cum.set_ylim(0) # Start y-axis at 0
else:
    ax_cum.text(0.5, 0.5, 'Не знайдено часів завершення успішних інференсів.', # Translated message
                horizontalalignment='center', verticalalignment='center', 
                transform=ax_cum.transAxes)

ax_cum.set_title(UKR_TRANSLATIONS['Scenario C: Cumulative Successful Inference Requests Over Time']) # Use translated title
ax_cum.set_xlabel(UKR_TRANSLATIONS['Time (seconds relative to first job start)']) # Use translated label
ax_cum.set_ylabel(UKR_TRANSLATIONS['Cumulative Successful Inference Requests']) # Use translated label
ax_cum.legend(title=UKR_TRANSLATIONS["Strategy"]) # Use translated legend title
ax_cum.grid(True, linestyle='--', alpha=0.7)

plt.tight_layout()
plt.savefig(plot_cumulative_inf_path, bbox_inches='tight')
print(f"Cumulative inference requests plot saved to: {plot_cumulative_inf_path}")
plt.close(fig_cum)


print("Visualization script finished.") 