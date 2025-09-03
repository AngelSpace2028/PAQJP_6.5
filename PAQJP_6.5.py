import os
import sys
import math
import struct
import array
import random
import heapq
import binascii
import logging
import paq  # Python binding for PAQ9a (pip install paq)
import hashlib
from enum import Enum
from typing import List, Dict, Tuple, Optional

# === Configure Logging ===
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(), logging.FileHandler('compression_results.log')]
)

# === Constants ===
PROGNAME = "PAQJP_6_Smart"
HUFFMAN_THRESHOLD = 1024
PI_DIGITS_FILE = "pi_digits.txt"
PRIMES = [p for p in range(2, 256) if all(p % d != 0 for d in range(2, int(p**0.5)+1))]
MEM = 1 << 15

# === Dictionary file list ===
DICTIONARY_FILES = [
    "words_enwik8.txt", "eng_news_2005_1M-sentences.txt", "eng_news_2005_1M-words.txt",
    "eng_news_2005_1M-sources.txt", "eng_news_2005_1M-co_n.txt",
    "eng_news_2005_1M-co_s.txt", "eng_news_2005_1M-inv_so.txt",
    "eng_news_2005_1M-meta.txt", "Dictionary.txt",
    "the-complete-reference-html-css-fifth-edition.txt", "francais.txt", "espanol.txt", "deutsch.txt", "ukenglish.txt", "vertebrate-palaeontology-dict.txt"
]

# === DNA Encoding Table ===
DNA_ENCODING_TABLE = {
    'AAAA': 0b00000, 'AAAC': 0b00001, 'AAAG': 0b00010, 'AAAT': 0b00011,
    'AACC': 0b00100, 'AACG': 0b00101, 'AACT': 0b00110, 'AAGG': 0b00111,
    'AAGT': 0b01000, 'AATT': 0b01001, 'ACCC': 0b01010, 'ACCG': 0b01011,
    'ACCT': 0b01100, 'AGGG': 0b01101, 'AGGT': 0b01110, 'AGTT': 0b01111,
    'CCCC': 0b10000, 'CCCG': 0b10001, 'CCCT': 0b10010, 'CGGG': 0b10011,
    'CGGT': 0b10100, 'CGTT': 0b10101, 'GTTT': 0b10110, 'CTTT': 0b10111,
    'AAAAAAAA': 0b11000, 'CCCCCCCC': 0b11001, 'GGGGGGGG': 0b11010, 'TTTTTTTT': 0b11011,
    'A': 0b11100, 'C': 0b11101, 'G': 0b11110, 'T': 0b11111
}
DNA_DECODING_TABLE = {v: k for k, v in DNA_ENCODING_TABLE.items()}

# === Pi Digits Functions ===
def save_pi_digits(digits: List[int], filename: str = PI_DIGITS_FILE) -> bool:
    try:
        with open(filename, 'w') as f:
            f.write(','.join(str(d) for d in digits))
        logging.info(f"Saved {len(digits)} pi digits to {filename}")
        return True
    except Exception as e:
        logging.error(f"Failed to save pi digits to {filename}: {e}")
        return False

def load_pi_digits(filename: str = PI_DIGITS_FILE, expected_count: int = 3) -> Optional[List[int]]:
    try:
        if not os.path.isfile(filename):
            logging.warning(f"Pi digits file {filename} does not exist")
            return None
        with open(filename, 'r') as f:
            data = f.read().strip()
            if not data:
                logging.warning(f"Pi digits file {filename} is empty")
                return None
            digits = []
            for x in data.split(','):
                if not x.isdigit():
                    logging.warning(f"Invalid integer in {filename}: {x}")
                    return None
                d = int(x)
                if not (0 <= d <= 255):
                    logging.warning(f"Digit out of range in {filename}: {d}")
                    return None
                digits.append(d)
            if len(digits) != expected_count:
                logging.warning(f"Loaded {len(digits)} digits, expected {expected_count}")
                return None
            logging.info(f"Loaded {len(digits)} pi digits from {filename}")
            return digits
    except Exception as e:
        logging.error(f"Failed to load pi digits from {filename}: {e}")
        return None

def generate_pi_digits(num_digits: int = 3, filename: str = PI_DIGITS_FILE) -> List[int]:
    loaded_digits = load_pi_digits(filename, num_digits)
    if loaded_digits is not None:
        return loaded_digits
    try:
        from mpmath import mp
        mp.dps = num_digits
        pi_digits = [int(d) for d in mp.pi.digits(10)[0]]
        if len(pi_digits) != num_digits:
            logging.error(f"Generated {len(pi_digits)} digits, expected {num_digits}")
            raise ValueError("Incorrect number of pi digits generated")
        if not all(0 <= d <= 9 for d in pi_digits):
            logging.error("Generated pi digits contain invalid values")
            raise ValueError("Invalid pi digits generated")
        mapped_digits = [(d * 255 // 9) % 256 for d in pi_digits]
        save_pi_digits(mapped_digits, filename)
        return mapped_digits
    except Exception as e:
        logging.error(f"Failed to generate pi digits: {e}")
        fallback_digits = [3, 1, 4]
        mapped_fallback = [(d * 255 // 9) % 256 for d in fallback_digits[:num_digits]]
        logging.warning(f"Using {len(mapped_fallback)} fallback pi digits")
        save_pi_digits(mapped_fallback, filename)
        return mapped_fallback

PI_DIGITS = generate_pi_digits(3)

# === Helper Classes and Functions ===
class Filetype(Enum):
    DEFAULT = 0
    JPEG = 1
    TEXT = 3

class Node:
    def __init__(self, left=None, right=None, symbol=None):
        self.left = left
        self.right = right
        self.symbol = symbol

    def is_leaf(self):
        return self.left is None and self.right is None

def transform_with_prime_xor_every_3_bytes(data, repeat=100):
    transformed = bytearray(data)
    for prime in PRIMES:
        xor_val = prime if prime == 2 else max(1, math.ceil(prime * 4096 / 28672))
        for _ in range(repeat):
            for i in range(0, len(transformed), 3):
                transformed[i] ^= xor_val
    return bytes(transformed)

def transform_with_pattern_chunk(data, chunk_size=4):
    transformed = bytearray()
    for i in range(0, len(data), chunk_size):
        chunk = data[i:i + chunk_size]
        transformed.extend([b ^ 0xFF for b in chunk])
    return bytes(transformed)

def is_prime(n):
    if n < 2:
        return False
    if n == 2:
        return True
    if n % 2 == 0:
        return False
    for i in range(3, int(n ** 0.5) + 1, 2):
        if n % i == 0:
            return False
    return True

def find_nearest_prime_around(n):
    offset = 0
    while True:
        if is_prime(n - offset):
            return n - offset
        if is_prime(n + offset):
            return n + offset
        offset += 1

# === State Table ===
class StateTable:
    def __init__(self):
        self.table = [
            [1, 2, 0, 0], [3, 5, 1, 0], [4, 6, 0, 1], [7, 10, 2, 0],
            [8, 12, 1, 1], [9, 13, 1, 1], [11, 14, 0, 2], [15, 19, 3, 0],
            [16, 23, 2, 1], [17, 24, 2, 1], [18, 25, 2, 1], [20, 27, 1, 2],
            [21, 28, 1, 2], [22, 29, 1, 2], [26, 30, 0, 3], [31, 33, 4, 0],
            [32, 35, 3, 1], [32, 35, 3, 1], [32, 35, 3, 1], [32, 35, 3, 1],
            [34, 37, 2, 2], [34, 37, 2, 2], [34, 37, 2, 2], [34, 37, 2, 2],
            [34, 37, 2, 2], [34, 37, 2, 2], [36, 39, 1, 3], [36, 39, 1, 3],
            [36, 39, 1, 3], [36, 39, 1, 3], [38, 40, 0, 4], [41, 43, 5, 0],
            [42, 45, 4, 1], [42, 45, 4, 1], [44, 47, 3, 2], [44, 47, 3, 2],
            [46, 49, 2, 3], [46, 49, 2, 3], [48, 51, 1, 4], [48, 51, 1, 4],
            [50, 52, 0, 5], [53, 43, 6, 0], [54, 57, 5, 1], [54, 57, 5, 1],
            [56, 59, 4, 2], [56, 59, 4, 2], [58, 61, 3, 3], [58, 61, 3, 3],
            [60, 63, 2, 4], [60, 63, 2, 4], [62, 65, 1, 5], [62, 65, 1, 5],
            [50, 66, 0, 6], [67, 55, 7, 0], [68, 57, 6, 1], [68, 57, 6, 1],
            [70, 73, 5, 2], [70, 73, 5, 2], [72, 75, 4, 3], [72, 75, 4, 3],
            [74, 77, 3, 4], [74, 77, 3, 4], [76, 79, 2, 5], [76, 79, 2, 5],
            [62, 81, 1, 6], [62, 81, 1, 6], [64, 82, 0, 7], [83, 69, 8, 0],
            [84, 76, 7, 1], [84, 76, 7, 1], [86, 73, 6, 2], [86, 73, 6, 2],
            [44, 59, 5, 3], [44, 59, 5, 3], [58, 61, 4, 4], [58, 61, 4, 4],
            [60, 49, 3, 5], [60, 49, 3, 5], [76, 89, 2, 6], [76, 89, 2, 6],
            [78, 91, 1, 7], [78, 91, 1, 7], [80, 92, 0, 8], [93, 69, 9, 0],
            [94, 87, 8, 1], [94, 87, 8, 1], [96, 45, 7, 2], [96, 45, 7, 2],
            [48, 99, 2, 7], [48, 99, 2, 7], [88, 101, 1, 8], [88, 101, 1, 8],
            [80, 102, 0, 9], [103, 69, 10, 0], [104, 87, 9, 1], [104, 87, 9, 1],
            [106, 57, 8, 2], [106, 57, 8, 2], [62, 109, 2, 8], [62, 109, 2, 8],
            [88, 111, 1, 9], [88, 111, 1, 9], [80, 112, 0, 10], [113, 85, 11, 0],
            [114, 87, 10, 1], [114, 87, 10, 1], [116, 57, 9, 2], [116, 57, 9, 2],
            [62, 119, 2, 9], [62, 119, 2, 9], [88, 121, 1, 10], [88, 121, 1, 10],
            [90, 122, 0, 11], [123, 85, 12, 0], [124, 97, 11, 1], [124, 97, 11, 1],
            [126, 57, 10, 2], [126, 57, 10, 2], [62, 129, 2, 10], [62, 129, 2, 10],
            [98, 131, 1, 11], [98, 131, 1, 11], [90, 132, 0, 12], [133, 85, 13, 0],
            [134, 97, 12, 1], [134, 97, 12, 1], [136, 57, 11, 2], [136, 57, 11, 2],
            [62, 139, 2, 11], [62, 139, 2, 11], [98, 141, 1, 12], [98, 141, 1, 12],
            [90, 142, 0, 13], [143, 95, 14, 0], [144, 97, 13, 1], [144, 97, 13, 1],
            [68, 57, 12, 2], [68, 57, 12, 2], [62, 81, 2, 12], [62, 81, 2, 12],
            [98, 147, 1, 13], [98, 147, 1, 13], [100, 148, 0, 14], [149, 95, 15, 0],
            [150, 107, 14, 1], [150, 107, 14, 1], [108, 151, 1, 14], [108, 151, 1, 14],
            [100, 152, 0, 15], [153, 95, 16, 0], [154, 107, 15, 1], [108, 155, 1, 15],
            [100, 156, 0, 16], [157, 95, 17, 0], [158, 107, 16, 1], [108, 159, 1, 16],
            [100, 160, 0, 17], [161, 105, 18, 0], [162, 107, 17, 1], [108, 163, 1, 17],
            [110, 164, 0, 18], [165, 105, 19, 0], [166, 117, 18, 1], [118, 167, 1, 18],
            [110, 168, 0, 19], [169, 105, 20, 0], [170, 117, 19, 1], [118, 171, 1, 19],
            [110, 172, 0, 20], [173, 105, 21, 0], [174, 117, 20, 1], [118, 175, 1, 20],
            [110, 176, 0, 21], [177, 105, 22, 0], [178, 117, 21, 1], [118, 179, 1, 21],
            [120, 184, 0, 23], [185, 115, 24, 0], [186, 127, 23, 1], [128, 187, 1, 23],
            [120, 188, 0, 24], [189, 115, 25, 0], [190, 127, 24, 1], [128, 191, 1, 24],
            [120, 192, 0, 25], [193, 115, 26, 0], [194, 127, 25, 1], [128, 195, 1, 25],
            [120, 196, 0, 26], [197, 115, 27, 0], [198, 127, 26, 1], [128, 199, 1, 26],
            [120, 200, 0, 27], [201, 115, 28, 0], [202, 127, 27, 1], [128, 203, 1, 27],
            [120, 204, 0, 28], [205, 115, 29, 0], [206, 127, 28, 1], [128, 207, 1, 28],
            [120, 208, 0, 29], [209, 125, 30, 0], [210, 127, 29, 1], [128, 211, 1, 29],
            [130, 212, 0, 30], [213, 125, 31, 0], [214, 137, 30, 1], [138, 215, 1, 30],
            [130, 216, 0, 31], [217, 125, 32, 0], [218, 137, 31, 1], [138, 219, 1, 31],
            [130, 220, 0, 32], [221, 125, 33, 0], [222, 137, 32, 1], [138, 223, 1, 32],
            [130, 224, 0, 33], [225, 125, 34, 0], [226, 137, 33, 1], [138, 227, 1, 33],
            [130, 228, 0, 34], [229, 125, 35, 0], [230, 137, 34, 1], [138, 231, 1, 34],
            [130, 232, 0, 35], [233, 125, 36, 0], [234, 137, 35, 1], [138, 235, 1, 35],
            [130, 236, 0, 36], [237, 125, 37, 0], [238, 137, 36, 1], [138, 239, 1, 36],
            [130, 240, 0, 37], [241, 125, 38, 0], [242, 137, 37, 1], [138, 243, 1, 37],
            [130, 244, 0, 38], [245, 135, 39, 0], [246, 137, 38, 1], [138, 247, 1, 38],
            [140, 248, 0, 39], [249, 135, 40, 0], [250, 69, 39, 1], [80, 251, 1, 39],
            [140, 252, 0, 40], [249, 135, 41, 0], [250, 69, 40, 1], [80, 251, 1, 40],
            [140, 252, 0, 41]
        ]


# === Smart Compressor ===
class SmartCompressor:
    def __init__(self):
        self.dictionaries = self.load_dictionaries()

    def load_dictionaries(self):
        data = []
        for filename in DICTIONARY_FILES:
            if os.path.exists(filename):
                try:
                    with open(filename, "r", encoding="utf-8", errors="ignore") as f:
                        data.append(f.read())
                    logging.info(f"Loaded dictionary: {filename}")
                except Exception as e:
                    logging.warning(f"Could not read {filename}: {e}")
            else:
                logging.warning(f"Missing dictionary: {filename}")
        return data

    def compute_sha256(self, data):
        return hashlib.sha256(data).hexdigest()

    def compute_sha256_binary(self, data):
        return hashlib.sha256(data).digest()

    def find_hash_in_dictionaries(self, hash_hex):
        for filename in DICTIONARY_FILES:
            if not os.path.exists(filename):
                continue
            try:
                with open(filename, "r", encoding="utf-8", errors="ignore") as f:
                    for line in f:
                        if hash_hex in line:
                            logging.info(f"Hash {hash_hex[:16]}... found in {filename}")
                            return filename
            except Exception as e:
                logging.warning(f"Error searching {filename}: {e}")
        return None

    def generate_8byte_sha(self, data):
        try:
            return hashlib.sha256(data).digest()[:8]
        except Exception as e:
            logging.error(f"Failed to generate SHA: {e}")
            return None

    def paq_compress(self, data):
        if not data:
            logging.warning("paq_compress: Empty input, returning empty bytes")
            return b''
        try:
            if isinstance(data, bytearray):
                data = bytes(data)
            elif not isinstance(data, bytes):
                raise TypeError(f"Expected bytes or bytearray, got {type(data)}")
            compressed = paq.compress(data)
            logging.info("PAQ9a compression complete")
            return compressed
        except Exception as e:
            logging.error(f"PAQ9a compression failed: {e}")
            return None

    def paq_decompress(self, data):
        if not data:
            logging.warning("paq_decompress: Empty input, returning empty bytes")
            return b''
        try:
            decompressed = paq.decompress(data)
            logging.info("PAQ9a decompression complete")
            return decompressed
        except Exception as e:
            logging.error(f"PAQ9a decompression failed: {e}")
            return None

    def reversible_transform(self, data):
        logging.info("Applying XOR transform (0xAA)")
        transformed = bytes(b ^ 0xAA for b in data)
        logging.info("XOR transform complete")
        return transformed

    def reverse_reversible_transform(self, data):
        logging.info("Reversing XOR transform (0xAA)")
        transformed = bytes(b ^ 0xAA for b in data)
        logging.info("Reverse XOR transform complete")
        return transformed

    def compress(self, data, input_filename):
        if not data:
            logging.warning("SmartCompressor: Empty input, returning empty bytes")
            return b''
        hash_hex = self.compute_sha256(data)
        if any(x in input_filename.lower() for x in ["words", "lines", "sentence"]) and input_filename.lower().endswith(".paq"):
            found = self.find_hash_in_dictionaries(hash_hex)
            if found:
                sha = self.generate_8byte_sha(data)
                if sha:
                    logging.info(f"Found dictionary match for {input_filename}, returning 8-byte SHA")
                    return sha
        transformed = self.reversible_transform(data)
        compressed = self.paq_compress(transformed)
        if compressed is None or len(compressed) >= len(data):
            logging.warning("SmartCompressor: Compression failed or not efficient, returning original data")
            return data
        hash_binary = self.compute_sha256_binary(data)
        logging.info(f"SmartCompressor: Compressed size = {len(compressed) + len(hash_binary)} bytes")
        return hash_binary + compressed

    def decompress(self, data, input_filename):
        if not data:
            logging.warning("SmartCompressor: Empty input, returning empty bytes")
            return b''
        if len(data) == 8 and any(x in input_filename.lower() for x in ["words", "lines", "sentence"]) and input_filename.lower().endswith(".paq"):
            for filename in DICTIONARY_FILES:
                if not os.path.exists(filename):
                    continue
                try:
                    with open(filename, "r", encoding="utf-8", errors="ignore") as f:
                        original_data = f.read().encode('utf-8')
                        if self.generate_8byte_sha(original_data) == data:
                            logging.info(f"Decompressed {input_filename} using dictionary match")
                            return original_data
                except Exception as e:
                    logging.warning(f"Error checking dictionary {filename}: {e}")
        if len(data) < 32:
            logging.error("SmartCompressor: Data too short for hash + compressed data")
            return None
        hash_binary = data[:32]
        compressed = data[32:]
        decompressed = self.paq_decompress(compressed)
        if decompressed is None:
            logging.error("SmartCompressor: Decompression failed")
            return None
        original = self.reverse_reversible_transform(decompressed)
        if self.compute_sha256_binary(original) != hash_binary:
            logging.error("SmartCompressor: Hash verification failed")
            return None
        logging.info("SmartCompressor: Decompression successful")
        return original

# === PAQJP Compressor ===
class PAQJPCompressor:
    def __init__(self):
        self.seed_table = self.generate_seed_table()

    def generate_seed_table(self):
        random.seed(42)
        return [random.randint(0, 255) for _ in range(256)]

    def transform_01(self, data, repeat=100):
        return transform_with_prime_xor_every_3_bytes(data, repeat)

    def reverse_transform_01(self, data, repeat=100):
        return transform_with_prime_xor_every_3_bytes(data, repeat)

    def transform_03(self, data):
        return transform_with_pattern_chunk(data)

    def reverse_transform_03(self, data):
        return transform_with_pattern_chunk(data)

    def transform_04(self, data):
        transformed = bytearray()
        for i, b in enumerate(data):
            transformed.append((b - i) % 256)
        return bytes(transformed)

    def reverse_transform_04(self, data):
        transformed = bytearray()
        for i, b in enumerate(data):
            transformed.append((b + i) % 256)
        return bytes(transformed)

    def transform_05(self, data):
        transformed = bytearray()
        for b in data:
            transformed.append(((b << 3) | (b >> 5)) & 0xFF)
        return bytes(transformed)

    def reverse_transform_05(self, data):
        transformed = bytearray()
        for b in data:
            transformed.append(((b >> 3) | (b << 5)) & 0xFF)
        return bytes(transformed)

    def transform_06(self, data):
        random.seed(42)
        substitution = list(range(256))
        random.shuffle(substitution)
        transformed = bytearray()
        for b in data:
            transformed.append(substitution[b])
        return bytes(transformed)

    def reverse_transform_06(self, data):
        random.seed(42)
        substitution = list(range(256))
        random.shuffle(substitution)
        reverse_substitution = [0] * 256
        for i, v in enumerate(substitution):
            reverse_substitution[v] = i
        transformed = bytearray()
        for b in data:
            transformed.append(reverse_substitution[b])
        return bytes(transformed)

    def transform_07(self, data):
        transformed = bytearray(data)
        size = len(data) % 256
        for i in range(len(data)):
            transformed[i] ^= PI_DIGITS[i % len(PI_DIGITS)] ^ size
        return bytes(transformed)

    def reverse_transform_07(self, data):
        return self.transform_07(data)

    def transform_08(self, data):
        transformed = bytearray(data)
        for i in range(len(data)):
            nearest_prime = find_nearest_prime_around(data[i])
            transformed[i] ^= nearest_prime ^ PI_DIGITS[i % len(PI_DIGITS)]
        return bytes(transformed)

    def reverse_transform_08(self, data):
        return self.transform_08(data)

    def transform_09(self, data):
        transformed = bytearray(data)
        for i in range(len(data)):
            nearest_prime = find_nearest_prime_around(data[i])
            transformed[i] ^= nearest_prime ^ self.seed_table[i % 256] ^ PI_DIGITS[i % len(PI_DIGITS)]
        return bytes(transformed)

    def reverse_transform_09(self, data):
        return self.transform_09(data)

    def transform_10(self, data):
        transformed = bytearray(data)
        seed = 42
        for i in range(len(data)):
            if i > 0 and data[i-1] == ord('X') and data[i] == ord('1'):
                seed = (seed * 1103515245 + 12345) & 0x7FFFFFFF
                transformed[i] ^= (seed >> 16) & 0xFF
        return bytes(transformed)

    def reverse_transform_10(self, data):
        return self.transform_10(data)

    def transform_11(self, data):
        best_transformed = data
        best_size = len(data)
        for y in range(256):
            transformed = bytearray(data)
            for i in range(len(data)):
                transformed[i] = (transformed[i] + y) % 256
            compressed = self.paq_compress(transformed)
            if compressed and len(compressed) < best_size:
                best_transformed = transformed
                best_size = len(compressed)
        return bytes(best_transformed)

    def reverse_transform_11(self, data):
        best_transformed = data
        best_size = len(data)
        for y in range(256):
            transformed = bytearray(data)
            for i in range(len(data)):
                transformed[i] = (transformed[i] - y) % 256
            compressed = self.paq_compress(transformed)
            if compressed and len(compressed) < best_size:
                best_transformed = transformed
                best_size = len(compressed)
        return bytes(best_transformed)

    def transform_12(self, data):
        transformed = bytearray(data)
        a, b = 0, 1
        for i in range(len(data)):
            transformed[i] ^= (a & 0xFF)
            a, b = b, (a + b) % 256
        return bytes(transformed)

    def reverse_transform_12(self, data):
        return self.transform_12(data)

    def transform_13(self, data):
        if not self.seed_table:
            logging.warning("transform_13: Empty state table, returning original data")
            return data
        transformed = bytearray()
        underflow = bytearray()
        state = 0
        for b in data:
            if state >= len(self.seed_table):
                logging.warning(f"transform_13: State {state} exceeds table size, using modulo")
                state = state % len(self.seed_table)
            transformed.append((b - self.seed_table[state]) % 256)
            underflow.append(1 if b < self.seed_table[state] else 0)
            state = self.seed_table[state]
        return bytes(transformed + underflow)

    def reverse_transform_13(self, data):
        if not self.seed_table:
            logging.warning("reverse_transform_13: Empty state table, returning original data")
            return data
        if len(data) < 2:
            logging.error("reverse_transform_13: Data too short")
            return None
        half = len(data) // 2
        transformed = bytearray(data[:half])
        underflow = data[half:]
        state = 0
        for i in range(len(transformed)):
            if state >= len(self.seed_table):
                logging.warning(f"reverse_transform_13: State {state} exceeds table size, using modulo")
                state = state % len(self.seed_table)
            transformed[i] = (transformed[i] + self.seed_table[state] + (256 if underflow[i] else 0)) % 256
            state = self.seed_table[state]
        return bytes(transformed)

    def transform_14(self, data):
        transformed = bytearray(data)
        i = 0
        while i < len(data) - 1:
            if data[i] == ord('0') and data[i+1] == ord('1'):
                for prime in PRIMES:
                    transformed[i] ^= prime
                    transformed[i+1] ^= prime
                i += 2
            else:
                i += 1
        return bytes(transformed)

    def reverse_transform_14(self, data):
        return self.transform_14(data)

    def transform_genomecompress(self, data):
        transformed = bytearray()
        i = 0
        while i < len(data):
            for pattern, code in DNA_ENCODING_TABLE.items():
                if data[i:i+len(pattern)] == pattern.encode():
                    transformed.append(code)
                    i += len(pattern)
                    break
            else:
                logging.warning(f"transform_genomecompress: No match at position {i}, skipping byte")
                transformed.append(data[i])
                i += 1
        return bytes(transformed)

    def reverse_transform_genomecompress(self, data):
        transformed = bytearray()
        for b in data:
            if b in DNA_DECODING_TABLE:
                transformed.extend(DNA_DECODING_TABLE[b].encode())
            else:
                logging.warning(f"reverse_transform_genomecompress: Invalid code {b}, skipping")
                transformed.append(b)
        return bytes(transformed)

    def transform_dynamic(self, data, marker):
        transformed = bytearray(data)
        for i in range(len(data)):
            transformed[i] ^= self.seed_table[marker % len(self.seed_table)]
        return bytes(transformed)

    def reverse_transform_dynamic(self, data, marker):
        return self.transform_dynamic(data, marker)

    def paq_compress(self, data):
        if not data:
            logging.warning("paq_compress: Empty input, returning empty bytes")
            return b''
        try:
            if isinstance(data, bytearray):
                data = bytes(data)
            elif not isinstance(data, bytes):
                raise TypeError(f"Expected bytes or bytearray, got {type(data)}")
            compressed = paq.compress(data)
            logging.info("PAQ9a compression complete")
            return compressed
        except Exception as e:
            logging.error(f"PAQ9a compression failed: {e}")
            return None

    def paq_decompress(self, data):
        if not data:
            logging.warning("paq_decompress: Empty input, returning empty bytes")
            return b''
        try:
            decompressed = paq.decompress(data)
            logging.info("PAQ9a decompression complete")
            return decompressed
        except Exception as e:
            logging.error(f"PAQ9a decompression failed: {e}")
            return None

    def huffman_compress(self, data):
        if not data:
            logging.warning("huffman_compress: Empty input, returning empty bytes")
            return b''
        freq = [0] * 256
        for b in data:
            freq[b] += 1
        heap = [[weight, [symbol, b'']] for symbol, weight in enumerate(freq) if weight]
        heapq.heapify(heap)
        while len(heap) > 1:
            lo = heapq.heappop(heap)
            hi = heapq.heappop(heap)
            for pair in lo[1:]:
                pair[1] = b'0' + pair[1]
            for pair in hi[1:]:
                pair[1] = b'1' + pair[1]
            heapq.heappush(heap, [lo[0] + hi[0]] + lo[1:] + hi[1:])
        codes = dict(heap[0][1:])
        compressed = bytearray()
        bits = ''
        for b in data:
            bits += ''.join(str(x) for x in codes[b])
            while len(bits) >= 8:
                compressed.append(int(bits[:8], 2))
                bits = bits[8:]
        if bits:
            compressed.append(int(bits.ljust(8, '0'), 2))
        header = bytearray()
        for i, f in enumerate(freq):
            if f:
                header.extend([i, (f >> 8) & 0xFF, f & 0xFF])
        header.append(0)
        return bytes(header) + compressed

    def huffman_decompress(self, data):
        if not data:
            logging.warning("huffman_decompress: Empty input, returning empty bytes")
            return b''
        freq = [0] * 256
        i = 0
        while i < len(data) and data[i] != 0:
            symbol = data[i]
            if i + 2 >= len(data):
                logging.error("huffman_decompress: Invalid header")
                return None
            freq[symbol] = (data[i+1] << 8) | data[i+2]
            i += 3
        i += 1
        if i >= len(data):
            logging.error("huffman_decompress: No compressed data")
            return None
        compressed = data[i:]
        heap = [[weight, [symbol, b'']] for symbol, weight in enumerate(freq) if weight]
        heapq.heapify(heap)
        while len(heap) > 1:
            lo = heapq.heappop(heap)
            hi = heapq.heappop(heap)
            for pair in lo[1:]:
                pair[1] = b'0' + pair[1]
            for pair in hi[1:]:
                pair[1] = b'1' + pair[1]
            heapq.heappush(heap, [lo[0] + hi[0]] + lo[1:] + hi[1:])
        codes = dict((code, symbol) for symbol, code in heap[0][1:])
        decompressed = bytearray()
        bits = ''
        for b in compressed:
            bits += f'{b:08b}'
        current_code = ''
        for bit in bits:
            current_code += bit
            if current_code.encode() in codes:
                decompressed.append(codes[current_code.encode()])
                current_code = ''
        return bytes(decompressed)

    def compress_with_best_method(self, data, filetype, input_filename, mode="slow", force_transform=None):
        if not data:
            logging.warning("compress_with_best_method: Empty input, returning empty bytes")
            return b''
        if force_transform is not None:
            transform_func = getattr(self, f'transform_{force_transform:02d}', None)
            if force_transform == 0:
                transform_func = self.transform_genomecompress
            if transform_func is None:
                logging.error(f"Invalid forced transform: {force_transform}")
                return bytes([0]) + data
            transformed = transform_func(data)
            compressed = self.paq_compress(transformed)
            if compressed is None:
                logging.warning(f"Compression failed for transform {force_transform}, returning original data")
                return bytes([0]) + data
            logging.info(f"Forced transform {force_transform}: Compressed size = {len(compressed) + 1} bytes")
            return bytes([force_transform]) + compressed
        if len(data) < HUFFMAN_THRESHOLD:
            compressed = self.huffman_compress(data)
            logging.info(f"Huffman compression: Compressed size = {len(compressed) + 1} bytes")
            return bytes([4]) + compressed
        best_compressed = None
        best_size = len(data)
        best_marker = 0
        transforms_to_try = [0, 7, 8, 9, 12, 13, 14] if filetype == Filetype.TEXT else [0, 1, 2, 3, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
        if mode == "slow":
            transforms_to_try.extend(range(15, 256))
        for marker in transforms_to_try:
            try:
                if marker == 0:
                    transformed = self.transform_genomecompress(data)
                elif marker in range(1, 15):
                    transform_func = getattr(self, f'transform_{marker:02d}')
                    transformed = transform_func(data)
                else:
                    transformed = self.transform_dynamic(data, marker)
                compressed = self.paq_compress(transformed)
                if compressed and len(compressed) < best_size:
                    best_compressed = compressed
                    best_size = len(compressed)
                    best_marker = marker
            except Exception as e:
                logging.warning(f"Transform {marker} failed: {e}, skipping")
        if best_compressed is None:
            logging.warning("No transformation improved compression, returning original data")
            return bytes([0]) + data
        logging.info(f"Best transform: marker {best_marker}, Compressed size = {best_size + 1} bytes")
        return bytes([best_marker]) + best_compressed

    def test_all_transformations(self, data, filetype, input_filename):
        if not data:
            logging.warning("test_all_transformations: Empty input, skipping")
            return
        logging.info(f"Testing all transformations for {input_filename} (size: {len(data)} bytes)")
        results = []
        transforms_to_try = [0, 1, 2, 3, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14] + list(range(15, 256))
        if len(data) < HUFFMAN_THRESHOLD:
            compressed = self.huffman_compress(data)
            size = len(compressed) + 1 if compressed else len(data)
            logging.info(f"Huffman (marker 4): Compressed size = {size} bytes")
            results.append((4, size))
        for marker in transforms_to_try:
            try:
                if marker == 0:
                    transformed = self.transform_genomecompress(data)
                elif marker in range(1, 15):
                    transform_func = getattr(self, f'transform_{marker:02d}')
                    transformed = transform_func(data)
                else:
                    transformed = self.transform_dynamic(data, marker)
                compressed = self.paq_compress(transformed)
                size = len(compressed) + 1 if compressed else len(data)
                logging.info(f"Transform {marker}: Compressed size = {size} bytes")
                results.append((marker, size))
            except Exception as e:
                logging.error(f"Transform {marker} failed: {e}")
                results.append((marker, len(data)))
        results.sort(key=lambda x: x[1])
        logging.info("Transformation results (sorted by compressed size):")
        for marker, size in results:
            logging.info(f"Marker {marker}: {size} bytes")
        return results

    def compress(self, data, filetype, input_filename, mode="slow", force_transform=None):
        return self.compress_with_best_method(data, filetype, input_filename, mode, force_transform)

    def decompress(self, data, filetype, input_filename):
        if not data:
            logging.warning("decompress: Empty input, returning empty bytes")
            return b''
        if len(data) < 1:
            logging.error("decompress: Data too short for marker")
            return None
        marker = data[0]
        compressed = data[1:]
        if marker == 4:
            if len(data) < HUFFMAN_THRESHOLD:
                return self.huffman_decompress(compressed)
            logging.error("Huffman marker used for large data")
            return None
        decompressed = self.paq_decompress(compressed)
        if decompressed is None:
            logging.error("PAQ9a decompression failed")
            return None
        if marker == 0:
            return self.reverse_transform_genomecompress(decompressed)
        elif marker in range(1, 15):
            reverse_func = getattr(self, f'reverse_transform_{marker:02d}', None)
            if reverse_func is None:
                logging.error(f"No reverse transform for marker {marker}")
                return None
            return reverse_func(decompressed)
        elif marker in range(15, 256):
            return self.reverse_transform_dynamic(decompressed, marker)
        logging.error(f"Invalid marker: {marker}")
        return None

def get_filetype(input_filename):
    ext = os.path.splitext(input_filename)[1].lower()
    if ext in ['.jpg', '.jpeg']:
        return Filetype.JPEG
    if ext in ['.txt', '.csv', '.xml']:
        return Filetype.TEXT
    try:
        with open(input_filename, 'rb') as f:
            data = f.read(100)
            if data.startswith(b'\xFF\xD8'):
                return Filetype.JPEG
            if all(c < 128 for c in data):
                return Filetype.TEXT
    except Exception as e:
        logging.warning(f"Could not read {input_filename} for filetype detection: {e}")
    return Filetype.DEFAULT

def main():
    print(f"{PROGNAME} version 6.4")
    smart_compressor = SmartCompressor()
    paqjp_compressor = PAQJPCompressor()
    while True:
        print("\n1. Compress\n2. Decompress\n3. Test All Transformations\n4. Exit")
        choice = input("Enter choice: ").strip()
        if choice == '4':
            break
        if choice not in ['1', '2', '3']:
            print("Invalid choice")
            continue
        input_filename = input("Enter input filename: ").strip()
        if not os.path.exists(input_filename):
            print(f"File {input_filename} does not exist")
            continue
        output_filename = input("Enter output filename: ").strip()
        filetype = get_filetype(input_filename)
        logging.info(f"Detected filetype: {filetype.name}")
        try:
            with open(input_filename, 'rb') as f:
                data = f.read()
            if choice == '1':
                mode = input("Enter mode (fast/slow): ").strip().lower()
                if mode not in ['fast', 'slow']:
                    print("Invalid mode, using slow")
                    mode = 'slow'
                force_transform = input("Force transform (e.g., 1 for transform_01, 0 for genomecompress, leave blank for auto): ").strip()
                force_transform = int(force_transform) if force_transform.isdigit() else None
                compressed = paqjp_compressor.compress(data, filetype, input_filename, mode, force_transform)
                if compressed is None:
                    compressed = smart_compressor.compress(data, input_filename)
                if compressed:
                    with open(output_filename, 'wb') as f:
                        f.write(compressed)
                    print(f"Compressed to {output_filename} (size: {len(compressed)} bytes)")
                else:
                    print("Compression failed")
            elif choice == '2':
                decompressed = paqjp_compressor.decompress(data, filetype, input_filename)
                if decompressed is None:
                    decompressed = smart_compressor.decompress(data, input_filename)
                if decompressed:
                    with open(output_filename, 'wb') as f:
                        f.write(decompressed)
                    print(f"Decompressed to {output_filename} (size: {len(decompressed)} bytes)")
                else:
                    print("Decompression failed")
            elif choice == '3':
                paqjp_compressor.test_all_transformations(data, filetype, input_filename)
                smart_compressed = smart_compressor.compress(data, input_filename)
                logging.info(f"SmartCompressor: Compressed size = {len(smart_compressed)} bytes")
                print("Transformation test results logged to compression_results.log")
        except Exception as e:
            logging.error(f"Operation failed: {e}")
            print(f"Error: {e}")

if __name__ == "__main__":
    main()
