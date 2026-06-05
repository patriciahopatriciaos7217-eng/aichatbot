"""
Simple in-memory cache for responses
"""
from functools import lru_cache
import hashlib
import time

# Simple dictionary cache
_response_cache = {}
_cache_ttl = 300  # 5 minutes

def get_cache_key(question: str) -> str:
    """Generate cache key from question"""
    return hashlib.md5(question.lower().encode()).hexdigest()

def get_cached_response(question: str):
    """Get cached response if exists and not expired"""
    key = get_cache_key(question)
    if key in _response_cache:
        entry = _response_cache[key]
        if time.time() - entry['timestamp'] < _cache_ttl:
            return entry['response']
    return None

def cache_response(question: str, response: dict):
    """Cache a response"""
    key = get_cache_key(question)
    _response_cache[key] = {
        'response': response,
        'timestamp': time.time()
    }

def clear_cache():
    """Clear all cached responses"""
    global _response_cache
    _response_cache = {}