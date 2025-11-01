🚀 **GPU-Accelerated Password Rule Ranker**

The ranker.py script is a high-performance, GPU-accelerated tool designed to evaluate and rank Hashcat password generation rules. It uses PyOpenCL and NumPy to efficiently process massive wordlists and rule sets by leveraging parallel execution on the GPU.

It implements a dual-scoring system to assess rules based on both uniqueness (generating new, previously unseen passwords) and effectiveness (generating passwords already found in a known cracked list).

✨ **Features**

1. GPU Acceleration: Utilizes PyOpenCL for massive parallel rule application, significantly reducing runtime compared to CPU-only solutions.

2. Dual-Scoring Metrics: Ranks rules based on:

- *Uniqueness Score:* How many new passwords the rule generates (not present in the original wordlist).

- *Effectiveness Score:* How many passwords the rule generates that are present in a user-provided cracked list.

- *Optimized Output:* Generates a ranked CSV report and an automatically optimized, high-impact rule file containing the top K best-performing rules.

- *Memory Optimized:* Uses efficient FNV-1a hashing and bitfield hash maps to manage billions of password candidates with minimal VRAM overhead.

- *Robust Large Scale Handling:* Uses 64-bit NumPy counters to prevent overflow issues when processing results beyond 4.3 billion counts.

🛠️ **Prerequisites**

Before running the script, ensure you have the following installed:

Python 3.x

PyOpenCL: Requires a working OpenCL runtime on your system (e.g., NVIDIA CUDA, AMD ROCm, Intel OpenCL).

Required Python Packages:

```pip install pyopencl numpy tqdm```


⚙️ **Installation and Setup**

Clone the Repository (or save the script):

wget https://github.com/A113L/ranker.py/raw/refs/heads/main/ranker.py

*Verify OpenCL:* Ensure pyopencl is installed and can detect your GPU.

**Prepare Input Files:** You will need three files:

*Wordlist:* A large list of base words (e.g., rockyou.txt).

*Rules:* A standard Hashcat rule file (e.g., best64.rule).

*Cracked List:* A list of known cracked passwords (e.g., from a past successful audit).

🚀 **Usage**

The script is executed via the command line and requires five primary arguments:

```
python ranker.py \
    -w <path_to_wordlist> \
    -r <path_to_rules> \
    -c <path_to_cracked_list> \
    -o <path_for_output_csv> \
    -t <top_k_rules_to_save>
```
