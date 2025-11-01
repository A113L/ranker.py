import argparse
import csv
import itertools
import math
import os
import sys
import warnings
import numpy as np
import pyopencl as cl
from tqdm import tqdm

# --- WARNING FILTERS ---
warnings.filterwarnings("ignore", message="overflow encountered in scalar multiply")
try:
    warnings.filterwarnings("ignore", category=cl.CompilerWarning)
except AttributeError:
    # Handle case where pyopencl.CompilerWarning is not available
    pass
# -----------------------

# ====================================================================
# --- CONSTANTS CONFIGURATION (OPTIMIZED FOR MODERN GPUs) ---
# ====================================================================
MAX_WORD_LEN = 32
MAX_OUTPUT_LEN = MAX_WORD_LEN * 2
MAX_RULE_ARGS = 2
MAX_RULES_IN_BATCH = 128
LOCAL_WORK_SIZE = 256

# BATCH SIZE FOR WORDS
WORDS_PER_GPU_BATCH = 50000

# Global Uniqueness Map Parameters (Adjusted for 8GB VRAM)
GLOBAL_HASH_MAP_BITS = 35
GLOBAL_HASH_MAP_WORDS = 1 << (GLOBAL_HASH_MAP_BITS - 5)
GLOBAL_HASH_MAP_BYTES = GLOBAL_HASH_MAP_WORDS * np.uint32(4)
GLOBAL_HASH_MAP_MASK = (1 << (GLOBAL_HASH_MAP_BITS - 5)) - 1

# Cracked Password Map Parameters (Adjusted for 8GB VRAM)
CRACKED_HASH_MAP_BITS = 33
CRACKED_HASH_MAP_WORDS = 1 << (CRACKED_HASH_MAP_BITS - 5)
CRACKED_HASH_MAP_BYTES = CRACKED_HASH_MAP_WORDS * np.uint32(4)
CRACKED_HASH_MAP_MASK = (1 << (CRACKED_HASH_MAP_BITS - 5)) - 1

# Rule IDs (Unchanged)
START_ID_SIMPLE = 0
NUM_SIMPLE_RULES = 10
START_ID_TD = 10
NUM_TD_RULES = 20
START_ID_S = 30
NUM_S_RULES = 256 * 256
START_ID_A = 30 + NUM_S_RULES
NUM_A_RULES = 3 * 256
# ====================================================================

# --- KERNEL SOURCE (OpenCL C) ---
KERNEL_SOURCE = f"""
// FNV-1a Hash implementation in OpenCL
unsigned int fnv1a_hash_32(const unsigned char* data, unsigned int len) {{
    unsigned int hash = 2166136261U;
    for (unsigned int i = 0; i < len; i++) {{
        hash ^= data[i];
        hash *= 16777619U;
    }}
    return hash;
}}

__kernel void hash_map_init_kernel(
    __global unsigned int* global_hash_map,
    __global const unsigned int* base_hashes,
    const unsigned int num_hashes,
    const unsigned int map_mask)
{{
    unsigned int global_id = get_global_id(0);
    if (global_id >= num_hashes) return;

    unsigned int word_hash = base_hashes[global_id];

    // Hash map bitfield logic
    unsigned int map_index = (word_hash >> 5) & map_mask;
    unsigned int bit_index = word_hash & 31;
    unsigned int set_bit = (1U << bit_index);

    // Atomically set the bit in the global hash map
    atomic_or(&global_hash_map[map_index], set_bit);
}}


__kernel void bfs_kernel(
    __global const unsigned char* base_words_in,
    __global const unsigned int* rules_in,
    // Note: The counts themselves are 32-bit in the kernel for atomic operations,
    // but the host-side aggregation will be 64-bit to prevent overflow.
    __global unsigned int* rule_uniqueness_counts,
    __global unsigned int* rule_effectiveness_counts,
    __global unsigned int* global_hash_map,
    __global const unsigned int* cracked_hash_map,
    const unsigned int num_words,
    const unsigned int num_rules_in_batch,
    const unsigned int max_word_len,
    const unsigned int max_output_len,
    const unsigned int global_map_mask,
    const unsigned int cracked_map_mask)
{{
    unsigned int global_id = get_global_id(0);

    // Calculate base indices and exit if ID is redundant
    unsigned int word_per_rule_count = num_words * num_rules_in_batch;
    if (global_id >= word_per_rule_count) return;

    unsigned int word_idx = global_id / num_rules_in_batch;
    unsigned int rule_batch_idx = global_id % num_rules_in_batch;

    // --- Local Variable Setup (Constants from Python) ---
    unsigned int start_id_simple = {START_ID_SIMPLE};
    unsigned int end_id_simple = start_id_simple + {NUM_SIMPLE_RULES};
    unsigned int start_id_TD = {START_ID_TD};
    unsigned int end_id_TD = start_id_TD + {NUM_TD_RULES};
    unsigned int start_id_s = {START_ID_S};
    unsigned int end_id_s = start_id_s + {NUM_S_RULES};
    unsigned int start_id_A = {START_ID_A};
    unsigned int end_id_A = start_id_A + {NUM_A_RULES};

    // Private memory buffer for the transformed word
    unsigned char result_temp[2 * {MAX_WORD_LEN}];

    __global const unsigned char* current_word_ptr = base_words_in + word_idx * max_word_len;

    unsigned int rule_size_in_int = 2 + {MAX_RULE_ARGS};
    __global const unsigned int* current_rule_ptr_int = rules_in + rule_batch_idx * rule_size_in_int;

    unsigned int rule_id = current_rule_ptr_int[0];
    unsigned int rule_args_int = current_rule_ptr_int[1];

    // Find word length
    unsigned int word_len = 0;
    for (unsigned int i = 0; i < max_word_len; i++) {{
        if (current_word_ptr[i] == 0) {{
            word_len = i;
            break;
        }}
    }}

    if (word_len == 0 && rule_id < start_id_A ) {{
        return;
    }}

    unsigned int out_len = 0;
    bool changed_flag = false;

    // Zero out the temporary buffer
    for(unsigned int i = 0; i < max_output_len; i++) {{
        result_temp[i] = 0;
    }}

    // --- START: Rule Application Logic (Copy from original source) ---
    
    if (rule_id >= start_id_simple && rule_id < end_id_simple) {{
    
        switch(rule_id - start_id_simple) {{
            case 0: {{ // 'l' (lowercase)
                out_len = word_len;
                for (unsigned int i = 0; i < word_len; i++) {{
                    unsigned char c = current_word_ptr[i];
                    if (c >= 'A' && c <= 'Z') {{
                        result_temp[i] = c + 32;
                        changed_flag = true;
                    }} else {{
                        result_temp[i] = c;
                    }}
                }}
                break;
            }}
            case 1: {{ // 'u' (uppercase)
                out_len = word_len;
                for (unsigned int i = 0; i < word_len; i++) {{
                    unsigned char c = current_word_ptr[i];
                    if (c >= 'a' && c <= 'z') {{
                        result_temp[i] = c - 32;
                        changed_flag = true;
                    }} else {{
                        result_temp[i] = c;
                    }}
                }}
                break;
            }}
            case 2: {{ // 'c' (capitalize)
                out_len = word_len;
                if (word_len > 0) {{
                    if (current_word_ptr[0] >= 'a' && current_word_ptr[0] <= 'z') {{
                        result_temp[0] = current_word_ptr[0] - 32;
                        changed_flag = true;
                    }} else {{
                        result_temp[0] = current_word_ptr[0];
                    }}
                    for (unsigned int i = 1; i < word_len; i++) {{
                        unsigned char c = current_word_ptr[i];
                        if (c >= 'A' && c <= 'Z') {{
                            result_temp[i] = c + 32;
                            changed_flag = true;
                        }} else {{
                            result_temp[i] = c;
                        }}
                    }}
                }}
                break;
            }}
            case 3: {{ // 'C' (invert capitalize)
                out_len = word_len;
                if (word_len > 0) {{
                    if (current_word_ptr[0] >= 'A' && current_word_ptr[0] <= 'Z') {{
                        result_temp[0] = current_word_ptr[0] + 32;
                        changed_flag = true;
                    }} else {{
                        result_temp[0] = current_word_ptr[0];
                    }}
                    for (unsigned int i = 1; i < word_len; i++) {{
                        unsigned char c = current_word_ptr[i];
                        if (c >= 'a' && c <= 'z') {{
                            result_temp[i] = c - 32;
                            changed_flag = true;
                        }} else {{
                            result_temp[i] = c;
                        }}
                    }}
                }}
                break;
            }}
            case 4: {{ // 't' (toggle case)
                out_len = word_len;
                for (unsigned int i = 0; i < word_len; i++) {{
                    unsigned char c = current_word_ptr[i];
                    if (c >= 'a' && c <= 'z') {{
                        result_temp[i] = c - 32;
                        changed_flag = true;
                    }} else if (c >= 'A' && c <= 'Z') {{
                        result_temp[i] = c + 32;
                        changed_flag = true;
                    }} else {{
                        result_temp[i] = c;
                    }}
                }}
                break;
            }}
            case 5: {{ // 'r' (reverse)
                out_len = word_len;
                for (unsigned int i = 0; i < word_len; i++) {{
                    result_temp[i] = current_word_ptr[word_len - 1 - i];
                }}
                if (word_len > 1) {{
                    for (unsigned int i = 0; i < word_len; i++) {{
                        if (result_temp[i] != current_word_ptr[i]) {{
                            changed_flag = true;
                            break;
                        }}
                    }}
                }}
                break;
            }}
            case 6: {{ // 'k' (swap first two chars)
                out_len = word_len;
                for(unsigned int i=0; i<word_len; i++) result_temp[i] = current_word_ptr[i];
                if (word_len >= 2) {{
                    if (current_word_ptr[0] != current_word_ptr[1]) changed_flag = true;
                    result_temp[0] = current_word_ptr[1];
                    result_temp[1] = current_word_ptr[0];
                }}
                break;
            }}
            case 7: {{ // ':' (identity/no change)
                out_len = word_len;
                for(unsigned int i=0; i<word_len; i++) result_temp[i] = current_word_ptr[i];
                changed_flag = false;
                break;
            }}
            case 8: {{ // 'd' (duplicate)
                out_len = word_len * 2;
                if (out_len > max_output_len || out_len == 0) {{
                    out_len = 0;	
                    changed_flag = false;
                    break;
                }}
                for(unsigned int i=0; i<word_len; i++) {{
                    result_temp[i] = current_word_ptr[i];
                    result_temp[word_len+i] = current_word_ptr[i];
                }}
                changed_flag = true;
                break;
            }}
            case 9: {{ // 'f' (reflect: word + reverse(word))
                out_len = word_len * 2;
                if (out_len > max_output_len || out_len == 0) {{
                    out_len = 0;
                    changed_flag = false;
                    break;
                }}
                for(unsigned int i=0; i<word_len; i++) {{
                    result_temp[i] = current_word_ptr[i];
                    result_temp[word_len+i] = current_word_ptr[word_len-1-i];
                }}
                if (word_len > 0) changed_flag = true;
                break;
            }}
        }}
    }} else if (rule_id >= start_id_TD && rule_id < end_id_TD) {{ // T, D rules
    
        unsigned char operator_char = rule_args_int & 0xFF;
        unsigned int pos_char = (rule_args_int >> 8) & 0xFF;
    
        unsigned int pos_to_change;
        if (pos_char >= '0' && pos_char <= '9') {{
            pos_to_change = pos_char - '0';
        }} else {{
            pos_to_change = max_word_len + 1;
        }}

    
        if (operator_char == 'T') {{ // 'T' (toggle case at pos)
              out_len = word_len;
              for (unsigned int i = 0; i < word_len; i++) {{
                  result_temp[i] = current_word_ptr[i];
              }}
              if (pos_to_change < word_len) {{
                  unsigned char c = current_word_ptr[pos_to_change];
                  if (c >= 'a' && c <= 'z') {{
                      result_temp[pos_to_change] = c - 32;
                      changed_flag = true;
                  }} else if (c >= 'A' && c <= 'Z') {{
                      result_temp[pos_to_change] = c + 32;
                      changed_flag = true;
                  }}
              }}
        }} else if (operator_char == 'D') {{ // 'D' (delete char at pos)
            unsigned int out_idx = 0;
            if (pos_to_change < word_len) {{
                for (unsigned int i = 0; i < word_len; i++) {{
                    if (i != pos_to_change) {{
                        result_temp[out_idx++] = current_word_ptr[i];
                    }} else {{
                        changed_flag = true;
                    }}
                }}
            }} else {{
                for (unsigned int i = 0; i < word_len; i++) {{
                    result_temp[i] = current_word_ptr[i];
                }}
                out_idx = word_len;
                changed_flag = false;
            }}
            out_len = out_idx;
        }}
    }}
    else if (rule_id >= start_id_s && rule_id < end_id_s) {{ // 's' rules (substitute first)
        out_len = word_len;
        for(unsigned int i=0; i<word_len; i++) result_temp[i] = current_word_ptr[i];
    
        unsigned char find = rule_args_int & 0xFF;
        unsigned char replace = (rule_args_int >> 8) & 0xFF;
    
        for(unsigned int i = 0; i < word_len; i++) {{
            if (current_word_ptr[i] == find) {{
                result_temp[i] = replace;
                changed_flag = true;
            }}
        }}
    }} else if (rule_id >= start_id_A && rule_id < end_id_A) {{ // Group A rules
    
        unsigned char cmd = rule_args_int & 0xFF;
        unsigned char arg = (rule_args_int >> 8) & 0xFF;
    
        if (cmd != '@') {{
            for(unsigned int i=0; i<word_len; i++) result_temp[i] = current_word_ptr[i];
        }}
    
        if (cmd == '^') {{ // Prepend
            if (word_len + 1 > max_output_len) {{
                out_len = 0;
                changed_flag = false;
            }} else {{
                for(unsigned int i=word_len; i>0; i--) {{
                    result_temp[i] = current_word_ptr[i-1];
                }}
                result_temp[0] = arg;
                out_len = word_len + 1;
                changed_flag = true;
            }}
        }} else if (cmd == '$') {{ // Append
            if (word_len + 1 > max_output_len) {{
                out_len = 0;
                changed_flag = false;
            }} else {{
                out_len = word_len + 1;
                for(unsigned int i=0; i<word_len; i++) {{
                    result_temp[i] = current_word_ptr[i];
                }}
                result_temp[word_len] = arg;
                changed_flag = true;
            }}
        }} else if (cmd == '@') {{ // Delete all instances of char
            unsigned int temp_idx = 0;
            for(unsigned int i=0; i<word_len; i++) {{
                if (current_word_ptr[i] != arg) {{
                    result_temp[temp_idx++] = current_word_ptr[i];
                }} else {{
                    changed_flag = true;
                }}
            }}
            out_len = temp_idx;
        }}
    }}
    
    // --- END: Rule Application Logic (Copy from original source) ---


    // --- Dual-Uniqueness Logic on GPU (Modified) ---

    if (changed_flag) {{

        unsigned int word_hash = fnv1a_hash_32(result_temp, out_len);

        // 1. Check against the Base Wordlist (Uniqueness Score)
        unsigned int global_map_index = (word_hash >> 5) & global_map_mask;
        unsigned int bit_index = word_hash & 31;
        unsigned int check_bit = (1U << bit_index);

        __global unsigned int* global_map_ptr = (__global unsigned int*)&global_hash_map[global_map_index];

        // Atomically read the value
        unsigned int current_global_word = atomic_and(global_map_ptr, 0xFFFFFFFFU);

        // If the word IS NOT in the base wordlist
        if (!(current_global_word & check_bit)) {{
            // Increment the Uniqueness Score for this rule.
            atomic_inc(&rule_uniqueness_counts[rule_batch_idx]);

            // 2. Check against the Cracked List (Effectiveness Score)
            unsigned int cracked_map_index = (word_hash >> 5) & cracked_map_mask;
            __global const unsigned int* cracked_map_ptr = (__global const unsigned int*)&cracked_hash_map[cracked_map_index];

            // Read the cracked map value (No atomic read needed as it's READ_ONLY)
            unsigned int current_cracked_word = *cracked_map_ptr;

            // If the word IS in the cracked list (i.e., we found a "true crack")
            if (current_cracked_word & check_bit) {{
                // Increment the Effectiveness Score for this rule.
                atomic_inc(&rule_effectiveness_counts[rule_batch_idx]);
            }}
        }}

    }} else {{
        return;
    }}
}}
"""

# --- HELPER FUNCTIONS (Python) ---

def fnv1a_hash_32_cpu(data):
    """Calculates FNV-1a hash for a byte array."""
    hash_val = np.uint32(2166136261)
    for byte in data:
        hash_val ^= np.uint32(byte)
        hash_val *= np.uint32(16777619)
    return hash_val

def get_word_count(path):
    """Counts total words in file to set up progress bar."""
    print(f"Counting words in: {path}...")
    count = 0
    try:
        with open(path, 'r', encoding='latin-1', errors='ignore') as f:
            count = sum(1 for line in f if line.strip())
    except FileNotFoundError:
        print(f"Error: Wordlist file not found at: {path}")
        sys.exit(1) 

    print(f"Total words found: {count:,}")
    return count

def load_rules(path):
    """Loads Hashcat rules from file."""
    print(f"Loading rules from: {path}...")
    rules_list = []
    rule_id_counter = 0
    try:
        with open(path, 'r', encoding='latin-1') as f:
            for line in f:
                rule = line.strip()
                if not rule or rule.startswith('#'):
                    continue
                rules_list.append({'rule_data': rule, 'rule_id': rule_id_counter, 'uniqueness_score': 0, 'effectiveness_score': 0})
                rule_id_counter += 1
    except FileNotFoundError:
        print(f"Error: Rules file not found at: {path}")
        sys.exit(1)

    print(f"Loaded {len(rules_list)} rules.")
    return rules_list

def load_cracked_hashes(path, max_len):
    """Loads a list of cracked passwords and returns their FNV-1a hashes."""
    print(f"Loading cracked list for effectiveness check from: {path}...")
    cracked_hashes = []
    try:
        with open(path, 'r', encoding='latin-1', errors='ignore') as f:
            for line in f:
                word = line.strip().encode('latin-1')
                if 1 <= len(word) <= max_len:
                    cracked_hashes.append(fnv1a_hash_32_cpu(word))
    except FileNotFoundError:
        print(f"Warning: Cracked list file not found at: {path}. Effectiveness scores will be zero.")
        return np.array([], dtype=np.uint32)

    unique_hashes = np.unique(np.array(cracked_hashes, dtype=np.uint32))
    print(f"Loaded {len(unique_hashes):,} unique cracked password hashes.")
    return unique_hashes


def encode_rule(rule_str, rule_id, max_args):
    """Encodes a rule as an array of uint32: [rule ID, arguments]"""
    rule_size_in_int = 2 + max_args
    encoded = np.zeros(rule_size_in_int, dtype=np.uint32)
    encoded[0] = np.uint32(rule_id)
    rule_chars = rule_str.encode('latin-1')
    args_int = 0
    if len(rule_chars) >= 1:
        args_int |= np.uint32(rule_chars[0])
    if len(rule_chars) >= 2:
        args_int |= (np.uint32(rule_chars[1]) << 8)
    encoded[1] = args_int
    return encoded

def save_ranking_data(ranking_list, output_csv_path):
    """
    Saves the scoring and ranking data to a separate CSV file.
    """
    
    print(f"\nSaving rule ranking data to: {output_csv_path}...")

    # Calculate a combined score for ranking: Effectiveness is 10x more important than Uniqueness
    for rule in ranking_list:
        rule['combined_score'] = rule.get('effectiveness_score', 0) * 10 + rule.get('uniqueness_score', 0)

    # Filtering and sorting
    ranked_rules = [rule for rule in ranking_list if rule.get('combined_score', 0) > 0]
    ranked_rules.sort(key=lambda rule: rule['combined_score'], reverse=True)

    if not ranked_rules:
        print("❌ No rules had a positive combined score. Ranking file not created.")
        return None

    try:
        with open(output_csv_path, 'w', newline='', encoding='utf-8') as f:
            fieldnames = ['Rank', 'Combined_Score', 'Effectiveness_Score', 'Uniqueness_Score', 'Rule_Data']
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

            for rank, rule in enumerate(ranked_rules, 1):
                writer.writerow({
                    'Rank': rank,
                    # Write score as a float string, which is the source of the error on reading
                    'Combined_Score': f"{rule['combined_score']}", 
                    'Effectiveness_Score': rule.get('effectiveness_score', 0),
                    'Uniqueness_Score': rule.get('uniqueness_score', 0),
                    'Rule_Data': rule['rule_data']
                })

        print(f"✅ Ranking data saved successfully to {output_csv_path}.")
        return output_csv_path
    except Exception as e:
        print(f"❌ Error while saving ranking data to CSV file: {e}")
        return None

def load_and_save_optimized_rules(csv_path, output_rules_path, top_k):
    """
    Loads ranking data from a CSV, re-sorts by Combined_Score, and saves the 
    Top K rules to a new rule file. Pre-pends the identity rule ':'.
    
    FIX: The 'Combined_Score' is now converted to a float before being converted to an int.
    """
    if not csv_path:
        print("\nOptimization skipped: Ranking CSV path is missing.")
        return

    print(f"\nLoading ranking from CSV: {csv_path} and saving Top {top_k} Optimized Rules to: {output_rules_path}...")
    
    ranked_data = []
    try:
        with open(csv_path, 'r', encoding='utf-8', errors='ignore') as f:
            reader = csv.DictReader(f)
            for row in reader:
                # --- FIX APPLIED HERE ---
                # 1. Convert the score string (which might be in scientific notation) to a float.
                # 2. Convert the float to an integer (using int() truncates, which is fine for a score).
                try:
                    score_float = float(row['Combined_Score'])
                    row['combined_score'] = int(score_float)
                except ValueError as e:
                    print(f"❌ Error converting score '{row['Combined_Score']}' to number. Skipping this rule.")
                    print(f"Original error: {e}")
                    continue
                # ------------------------
                
                ranked_data.append(row)
    except FileNotFoundError:
        print(f"❌ Error: Ranking CSV file not found at: {csv_path}")
        return
    except Exception as e:
        print(f"❌ Error while reading CSV: {e}")
        return

    ranked_data.sort(key=lambda row: row['combined_score'], reverse=True)
    final_optimized_list = ranked_data[:top_k]

    if not final_optimized_list:
        print("❌ No rules available after sorting/filtering. Optimized rule file not created.")
        return

    try:
        with open(output_rules_path, 'w', newline='\n', encoding='utf-8') as f:
            # 1. Write the mandatory identity rule
            f.write(":\n")
            
            # 2. Write the top-ranked rules
            rules_written = 0
            for rule in final_optimized_list:
                if rule['Rule_Data'] != ':':
                    f.write(f"{rule['Rule_Data']}\n")
                    rules_written += 1
                    
        final_count = rules_written + 1
        print(f"✅ Top {final_count} optimized rules (including the prepended ':' rule) saved successfully to {output_rules_path}.")
    except Exception as e:
        print(f"❌ Error while saving optimized rules to file: {e}")


def wordlist_iterator(wordlist_path, max_len, batch_size):
    """
    Generator that yields batches of words and initial hashes directly from disk.
    """
    base_words_np = np.zeros((batch_size, max_len), dtype=np.uint8)
    base_hashes = []
    current_batch_count = 0

    try:
        with open(wordlist_path, 'r', encoding='latin-1', errors='ignore') as f:
            for line in f:
                word_str = line.strip()
                word = word_str.encode('latin-1')

                if 1 <= len(word) <= max_len:

                    base_words_np[current_batch_count, :len(word)] = np.frombuffer(word, dtype=np.uint8)
                    base_hashes.append(fnv1a_hash_32_cpu(word))

                    current_batch_count += 1

                    if current_batch_count == batch_size:
                        # Return flattened array, count, and hashes
                        yield base_words_np.ravel().copy(), current_batch_count, np.array(base_hashes, dtype=np.uint32)

                        base_words_np.fill(0)
                        base_hashes = []
                        current_batch_count = 0

            if current_batch_count > 0:
                words_to_yield = base_words_np[:current_batch_count * max_len].ravel().copy()
                yield words_to_yield, current_batch_count, np.array(base_hashes, dtype=np.uint32)
    except Exception as e:
        print(f"Error during wordlist iteration: {e}")


# --- SINGLE PROCESS RANKING FUNCTION ---

def rank_all_rules_single_process(wordlist_path, cracked_hashes_np, all_rules):
    """
    Single-threaded function that initializes OpenCL and iterates through the
    wordlist and all rules to perform ranking.
    """
    
    # 1. OpenCL Initialization (SINGLE INSTANCE)
    print("--- OpenCL Initialization ---")
    try:
        platform = cl.get_platforms()[0]
        devices = platform.get_devices(cl.device_type.GPU)
        if not devices:
            devices = platform.get_devices(cl.device_type.ALL) 
        if not devices:
            print("Error: No OpenCL device found.")
            return []
            
        device = devices[0]
        context = cl.Context([device])
        queue = cl.CommandQueue(context)

        rule_size_in_int = 2 + MAX_RULE_ARGS

        prg = cl.Program(context, KERNEL_SOURCE).build()
        kernel_bfs = prg.bfs_kernel
        kernel_init = prg.hash_map_init_kernel
        print(f"OpenCL initialized on device: {device.name.strip()}")
    except cl.Error as e:
        print(f"OpenCL Error during initialization: {e}")
        return []
        
    print("--- Data Preparation ---")
    # Encode all rules
    encoded_rules = [encode_rule(rule['rule_data'], rule['rule_id'], MAX_RULE_ARGS) for rule in all_rules]
    total_rules = len(all_rules)
    
    # 2. OpenCL Buffer Setup
    mf = cl.mem_flags

    base_words_size = WORDS_PER_GPU_BATCH * MAX_WORD_LEN * np.uint8().itemsize
    base_words_in_g = cl.Buffer(context, mf.READ_ONLY, base_words_size)
    base_hashes_size = WORDS_PER_GPU_BATCH * np.uint32().itemsize
    base_hashes_g = cl.Buffer(context, mf.READ_ONLY, base_hashes_size)

    rules_np_batch = np.zeros(MAX_RULES_IN_BATCH * rule_size_in_int, dtype=np.uint32)
    rules_in_g = cl.Buffer(context, mf.READ_ONLY, rules_np_batch.nbytes)

    global_hash_map_g = cl.Buffer(context, mf.READ_WRITE, GLOBAL_HASH_MAP_BYTES)

    cracked_hash_map_np = np.zeros(CRACKED_HASH_MAP_WORDS, dtype=np.uint32)
    cracked_hash_map_g = cl.Buffer(context, mf.READ_ONLY, cracked_hash_map_np.nbytes)

    rule_uniqueness_counts_np = np.zeros(MAX_RULES_IN_BATCH, dtype=np.uint32) 
    rule_effectiveness_counts_np = np.zeros(MAX_RULES_IN_BATCH, dtype=np.uint32) 
    rule_uniqueness_counts_g = cl.Buffer(context, mf.READ_WRITE, rule_uniqueness_counts_np.nbytes)
    rule_effectiveness_counts_g = cl.Buffer(context, mf.READ_WRITE, rule_effectiveness_counts_np.nbytes)

    # 3. INITIALIZE CRACKED HASH MAP (ONCE)
    if cracked_hashes_np.size > 0:
        cracked_temp_g = cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=cracked_hashes_np)
        global_size_init_cracked = (int(math.ceil(cracked_hashes_np.size / LOCAL_WORK_SIZE)) * LOCAL_WORK_SIZE,)
        local_size_init_cracked = (LOCAL_WORK_SIZE,)

        kernel_init(queue, global_size_init_cracked, local_size_init_cracked,
                    cracked_hash_map_g,
                    cracked_temp_g,
                    np.uint32(cracked_hashes_np.size),
                    np.uint32(CRACKED_HASH_MAP_MASK)).wait()
        print("Cracked hash map initialized on GPU.")

    # Host-side aggregation (one entry per rule)
    final_uniqueness_agg = np.zeros(total_rules, dtype=np.uint64)
    final_effectiveness_agg = np.zeros(total_rules, dtype=np.uint64)

    # 4. Disk-Based Ranking Loop
    print("\n--- Starting Ranking Loop ---")
    
    try:
        total_word_count = get_word_count(wordlist_path)
        total_batches = math.ceil(total_word_count / WORDS_PER_GPU_BATCH)
    except SystemExit:
        return []
    
    word_iterator = wordlist_iterator(wordlist_path, MAX_WORD_LEN, WORDS_PER_GPU_BATCH)
    pbar = tqdm(word_iterator, total=total_batches, desc="GPU Ranking", unit=" batch")

    for base_words_np_batch, num_words_batch, base_hashes_np_batch in pbar:

        # A. Initialize Base Word Hash Map with current word batch (ON GPU)
        cl.enqueue_fill_buffer(queue, global_hash_map_g, np.uint32(0), 0, GLOBAL_HASH_MAP_BYTES).wait()
        cl.enqueue_copy(queue, base_hashes_g, base_hashes_np_batch).wait()

        global_size_init = (int(math.ceil(num_words_batch / LOCAL_WORK_SIZE)) * LOCAL_WORK_SIZE,)
        local_size_init = (LOCAL_WORK_SIZE,)

        kernel_init(queue, global_size_init, local_size_init,
                    global_hash_map_g,
                    base_hashes_g,
                    np.uint32(num_words_batch),
                    np.uint32(GLOBAL_HASH_MAP_MASK)).wait()

        cl.enqueue_copy(queue, base_words_in_g, base_words_np_batch).wait()

        # B. Iterate over rule batches (all rules in one process)
        rule_starts = list(range(0, total_rules, MAX_RULES_IN_BATCH))
        
        for rule_idx_start in rule_starts:

            rule_idx_end = min(rule_idx_start + MAX_RULES_IN_BATCH, total_rules)
            current_batch_size = rule_idx_end - rule_idx_start

            # B1. Prepare and update rule and counter buffers for the GPU
            rules_np_batch.fill(0)
            
            for i in range(current_batch_size):
                encoded_rule = encoded_rules[rule_idx_start + i]
                rules_np_batch[i * rule_size_in_int : (i + 1) * rule_size_in_int] = encoded_rule
            
            cl.enqueue_copy(queue, rules_in_g, rules_np_batch).wait()
            cl.enqueue_fill_buffer(queue, rule_uniqueness_counts_g, np.uint32(0), 0, rule_uniqueness_counts_np.nbytes).wait()
            cl.enqueue_fill_buffer(queue, rule_effectiveness_counts_g, np.uint32(0), 0, rule_effectiveness_counts_np.nbytes).wait()


            # B2. Execute the BFS kernel
            global_size_bfs = (num_words_batch * current_batch_size,)
            
            # Pad global size to be a multiple of local work size
            global_size_padded = (int(math.ceil(global_size_bfs[0] / LOCAL_WORK_SIZE)) * LOCAL_WORK_SIZE,)
            
            kernel_bfs(queue, global_size_padded, (LOCAL_WORK_SIZE,),
                       base_words_in_g,
                       rules_in_g,
                       rule_uniqueness_counts_g,
                       rule_effectiveness_counts_g,
                       global_hash_map_g,
                       cracked_hash_map_g,
                       np.uint32(num_words_batch),
                       np.uint32(current_batch_size),
                       np.uint32(MAX_WORD_LEN),
                       np.uint32(MAX_OUTPUT_LEN),
                       np.uint32(GLOBAL_HASH_MAP_MASK),
                       np.uint32(CRACKED_HASH_MAP_MASK)).wait()

            # B3. Read results and aggregate on host (CPU)
            cl.enqueue_copy(queue, rule_uniqueness_counts_np, rule_uniqueness_counts_g).wait()
            cl.enqueue_copy(queue, rule_effectiveness_counts_np, rule_effectiveness_counts_g).wait()

            # Aggregate scores for the full rule list
            for i in range(current_batch_size):
                global_rule_index = rule_idx_start + i
                final_uniqueness_agg[global_rule_index] += rule_uniqueness_counts_np[i]
                final_effectiveness_agg[global_rule_index] += rule_effectiveness_counts_np[i]
                
    # 5. Attach final scores to rule objects
    for i, rule in enumerate(all_rules):
        rule['uniqueness_score'] = final_uniqueness_agg[i]
        rule['effectiveness_score'] = final_effectiveness_agg[i]

    print("\n--- Ranking Complete ---")
    return all_rules


# --- MAIN EXECUTION BLOCK ---

def main():
    parser = argparse.ArgumentParser(description="OpenCL-accelerated Hashcat Rule Ranker (Single-Threaded).")
    parser.add_argument('-w', '--wordlist', required=True, help="Path to the base wordlist file.")
    parser.add_argument('-r', '--rules', required=True, help="Path to the Hashcat rules file (e.g., 'best64.rule').")
    parser.add_argument('-c', '--cracked', required=True, help="Path to the list of known cracked passwords (for effectiveness score).")
    parser.add_argument('-o', '--output-base', default='rule_ranking', help="Base filename for output files. Creates <BASE>.csv and <BASE>_optimized.rule.")
    parser.add_argument('-t', '--top-k', type=int, default=10000, help="Number of top rules to save to the optimized rule file.")

    if len(sys.argv) == 1:
        parser.print_help(sys.stderr)
        sys.exit(1)

    args = parser.parse_args()

    # Define output file paths based on the new output-base argument
    output_csv_path = f"{args.output_base}.csv"
    output_rules_path = f"{args.output_base}_optimized.rule"

    # 1. Load Data
    all_rules = load_rules(args.rules)
    
    # Exit if no rules loaded
    if not all_rules:
        sys.exit(0)
        
    # The max length for passwords we check is MAX_OUTPUT_LEN
    cracked_hashes_np = load_cracked_hashes(args.cracked, MAX_OUTPUT_LEN)

    # 2. Start Single-Process Ranking
    final_ranked_rules = rank_all_rules_single_process(args.wordlist, cracked_hashes_np, all_rules)
    
    if not final_ranked_rules:
        print("Exiting due to an error during the ranking process.")
        sys.exit(1)

    # 3. Save Results
    output_csv_final_path = save_ranking_data(final_ranked_rules, output_csv_path)

    # 4. Create Optimized Rule File
    load_and_save_optimized_rules(output_csv_final_path, output_rules_path, args.top_k)


if __name__ == '__main__':
    main()
