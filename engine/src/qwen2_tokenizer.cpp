#include "brahma/qwen2_tokenizer.hpp"
#include <algorithm>
#include <cctype>
#include <limits>
#include <stdexcept>
#include <utility>
#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <Windows.h>
#endif

namespace brahma {
namespace {
void append_utf8(std::string& out, std::uint32_t cp) {
    if (cp <= 0x7f) out.push_back(static_cast<char>(cp));
    else if (cp <= 0x7ff) {
        out.push_back(static_cast<char>(0xc0 | (cp >> 6)));
        out.push_back(static_cast<char>(0x80 | (cp & 0x3f)));
    } else if (cp <= 0xffff) {
        out.push_back(static_cast<char>(0xe0 | (cp >> 12)));
        out.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3f)));
        out.push_back(static_cast<char>(0x80 | (cp & 0x3f)));
    } else {
        out.push_back(static_cast<char>(0xf0 | (cp >> 18)));
        out.push_back(static_cast<char>(0x80 | ((cp >> 12) & 0x3f)));
        out.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3f)));
        out.push_back(static_cast<char>(0x80 | (cp & 0x3f)));
    }
}
std::uint32_t read_cp(const std::string& s, std::size_t& i) {
    const auto c = static_cast<unsigned char>(s[i++]);
    if (c < 0x80) return c;
    unsigned extra = c >= 0xf0 ? 3u : c >= 0xe0 ? 2u : 1u;
    std::uint32_t cp = c & (extra == 3 ? 0x07u : extra == 2 ? 0x0fu : 0x1fu);
    for (unsigned j = 0; j < extra && i < s.size(); ++j)
        cp = (cp << 6) | (static_cast<unsigned char>(s[i++]) & 0x3fu);
    return cp;
}
const std::vector<std::uint32_t>& byte_to_codepoint() {
    static const auto table = [] {
        std::vector<std::uint32_t> map(256);
        std::vector<bool> direct(256, false);
        for (int b = 33; b <= 126; ++b) direct[static_cast<std::size_t>(b)] = true;
        for (int b = 161; b <= 172; ++b) direct[static_cast<std::size_t>(b)] = true;
        for (int b = 174; b <= 255; ++b) direct[static_cast<std::size_t>(b)] = true;
        std::uint32_t n = 0;
        for (std::size_t b = 0; b < 256; ++b) {
            if (direct[b]) map[b] = static_cast<std::uint32_t>(b);
            else map[b] = 256u + n++;
        }
        return map;
    }();
    return table;
}
const std::unordered_map<std::uint32_t, std::uint8_t>& codepoint_to_byte() {
    static const auto table = [] {
        std::unordered_map<std::uint32_t, std::uint8_t> map;
        const auto& forward = byte_to_codepoint();
        for (std::size_t i = 0; i < forward.size(); ++i)
            map.emplace(forward[i], static_cast<std::uint8_t>(i));
        return map;
    }();
    return table;
}
std::string byte_symbol(unsigned char b) {
    std::string out;
    append_utf8(out, byte_to_codepoint()[b]);
    return out;
}
struct CodePoint {
    std::uint32_t value;
    std::size_t begin;
    std::size_t end;
};

std::vector<CodePoint> codepoints(const std::string& text) {
    std::vector<CodePoint> result;
    for (std::size_t i = 0; i < text.size();) {
        const auto begin = i;
        const auto cp = read_cp(text, i);
        result.push_back({cp, begin, i});
    }
    return result;
}

bool is_line_break(std::uint32_t cp) { return cp == '\r' || cp == '\n'; }

bool is_unicode_space(std::uint32_t cp) {
    if (cp == 0x20 || (cp >= 0x09 && cp <= 0x0d) || cp == 0x85 ||
        cp == 0xa0 || cp == 0x1680 || (cp >= 0x2000 && cp <= 0x200a) ||
        cp == 0x2028 || cp == 0x2029 || cp == 0x202f || cp == 0x205f ||
        cp == 0x3000) return true;
#ifdef _WIN32
    if (cp <= 0xffff) {
        const wchar_t ch = static_cast<wchar_t>(cp);
        WORD type = 0;
        if (GetStringTypeW(CT_CTYPE1, &ch, 1, &type) != 0 && (type & C1_SPACE) != 0)
            return true;
    }
#endif
    return false;
}

bool is_unicode_letter_or_number(std::uint32_t cp) {
    if ((cp >= 'a' && cp <= 'z') || (cp >= 'A' && cp <= 'Z') ||
        (cp >= '0' && cp <= '9')) return true;
#ifdef _WIN32
    if (cp <= 0xffff) {
        const wchar_t ch = static_cast<wchar_t>(cp);
        WORD type = 0;
        if (GetStringTypeW(CT_CTYPE1, &ch, 1, &type) != 0 &&
            (type & (C1_ALPHA | C1_DIGIT)) != 0) return true;
    }
#endif
    // Portable fallback for common scripts when Windows Unicode classification is unavailable.
    return (cp >= 0x00c0 && cp <= 0x02af) ||
           (cp >= 0x0370 && cp <= 0x052f) ||
           (cp >= 0x0531 && cp <= 0x1fff) ||
           (cp >= 0x2c00 && cp <= 0xd7ff) ||
           (cp >= 0xf900 && cp <= 0xfaff) ||
           (cp >= 0x10000 && cp <= 0x1efff) ||
           (cp >= 0x0660 && cp <= 0x0669) ||
           (cp >= 0x06f0 && cp <= 0x06f9) ||
           (cp >= 0x0966 && cp <= 0x096f) ||
           (cp >= 0xff10 && cp <= 0xff19);
}

bool is_unicode_letter(std::uint32_t cp) {
    if (cp >= '0' && cp <= '9') return false;
    if (cp >= 0x0660 && cp <= 0x0669) return false;
    if (cp >= 0x06f0 && cp <= 0x06f9) return false;
    if (cp >= 0x0966 && cp <= 0x096f) return false;
    if (cp >= 0xff10 && cp <= 0xff19) return false;
#ifdef _WIN32
    if (cp <= 0xffff) {
        const wchar_t ch = static_cast<wchar_t>(cp);
        WORD type = 0;
        if (GetStringTypeW(CT_CTYPE1, &ch, 1, &type) != 0)
            return (type & C1_ALPHA) != 0;
    }
#endif
    return is_unicode_letter_or_number(cp) &&
           !((cp >= 0x00d7 && cp <= 0x00f7) ||
             (cp >= 0x2000 && cp <= 0x206f));
}

bool is_ascii_contraction(const std::string& text, std::size_t byte_pos, std::size_t& byte_end) {
    if (byte_pos >= text.size() || text[byte_pos] != '\'') return false;
    static constexpr const char* endings[] = {"re", "ve", "ll", "s", "t", "m", "d"};
    for (const auto* ending : endings) {
        std::size_t j = 0;
        while (ending[j] != '\0' && byte_pos + 1 + j < text.size()) {
            const auto c = static_cast<unsigned char>(text[byte_pos + 1 + j]);
            if (static_cast<char>(std::tolower(c)) != ending[j]) break;
            ++j;
        }
        if (ending[j] == '\0') {
            byte_end = byte_pos + 1 + j;
            return true;
        }
    }
    return false;
}

// Mirrors the ordered Qwen2/GPT-2-family Unicode regex pre-tokenizer:
// contractions, optional prefix + letters, 1-3 digit groups, punctuation,
// newline groups, trailing whitespace, then remaining whitespace.
std::vector<std::string> pretokenize(const std::string& text) {
    const auto cps = codepoints(text);
    std::vector<std::string> out;
    std::size_t i = 0;
    const auto emit = [&](std::size_t first, std::size_t last) {
        if (first < last) out.emplace_back(text.substr(cps[first].begin, cps[last - 1].end - cps[first].begin));
    };
    while (i < cps.size()) {
        const auto start = i;
        std::size_t contraction_end = 0;
        if (cps[i].value == '\'' && is_ascii_contraction(text, cps[i].begin, contraction_end)) {
            while (i < cps.size() && cps[i].end <= contraction_end) ++i;
            emit(start, i);
            continue;
        }

        // [^\r\n\p{L}\p{N}]?\p{L}+
        std::size_t letter_start = i;
        if (!is_unicode_letter_or_number(cps[i].value) && !is_line_break(cps[i].value) &&
            i + 1 < cps.size() && is_unicode_letter(cps[i + 1].value)) {
            letter_start = i + 1;
        }
        if (is_unicode_letter(cps[letter_start].value)) {
            i = letter_start + 1;
            while (i < cps.size() && is_unicode_letter(cps[i].value)) ++i;
            emit(start, i);
            continue;
        }

        // \p{N}{1,3}; longer digit runs are split into groups of at most 3.
        if (is_unicode_letter_or_number(cps[i].value) && !is_unicode_letter(cps[i].value)) {
            std::size_t count = 0;
            while (i < cps.size() && is_unicode_letter_or_number(cps[i].value) &&
                   !is_unicode_letter(cps[i].value) && count < 3) {
                ++i;
                ++count;
            }
            emit(start, i);
            continue;
        }

        //  ?[^\s\p{L}\p{N}]+[\r\n]*
        std::size_t punctuation_start = i;
        if (cps[i].value == ' ' && i + 1 < cps.size() &&
            !is_unicode_space(cps[i + 1].value) &&
            !is_unicode_letter_or_number(cps[i + 1].value)) {
            punctuation_start = i + 1;
        }
        if (!is_unicode_space(cps[punctuation_start].value) &&
            !is_unicode_letter_or_number(cps[punctuation_start].value)) {
            i = punctuation_start + 1;
            while (i < cps.size() && !is_unicode_space(cps[i].value) &&
                   !is_unicode_letter_or_number(cps[i].value)) ++i;
            while (i < cps.size() && is_line_break(cps[i].value)) ++i;
            emit(start, i);
            continue;
        }

        // \s*[\r\n]+
        if (is_line_break(cps[i].value) || is_unicode_space(cps[i].value)) {
            std::size_t j = i;
            while (j < cps.size() && is_unicode_space(cps[j].value) &&
                   !is_line_break(cps[j].value)) ++j;
            if (j < cps.size() && is_line_break(cps[j].value)) {
                i = j;
                while (i < cps.size() && is_line_break(cps[i].value)) ++i;
                emit(start, i);
                continue;
            }
            // \s+(?!\S) consumes trailing whitespace as one piece.
            bool trailing = true;
            for (std::size_t k = i; k < cps.size(); ++k) {
                if (!is_unicode_space(cps[k].value)) { trailing = false; break; }
            }
            if (trailing) {
                i = cps.size();
                emit(start, i);
                continue;
            }
            while (i < cps.size() && is_unicode_space(cps[i].value)) ++i;
            emit(start, i);
            continue;
        }

        // Defensive progress for unclassified code points.
        ++i;
        emit(start, i);
    }
    return out;
}
bool is_special(const std::string& s) {
    return s.size() >= 4 && s.rfind("<|", 0) == 0 && s.compare(s.size() - 2, 2, "|>") == 0;
}
std::string merge_key(const std::string& a, const std::string& b) {
    std::string k; k.reserve(a.size() + b.size() + 1); k += a; k.push_back('\x1f'); k += b; return k;
}
std::uint32_t metadata_id(const GgufModelInfo& model, const char* key, bool& found) {
    const auto it = model.metadata.find(key);
    if (it == model.metadata.end()) { found = false; return 0; }
    std::uint64_t value{};
    if (const auto* u = std::get_if<std::uint64_t>(&it->second)) value = *u;
    else if (const auto* s = std::get_if<std::int64_t>(&it->second); s && *s >= 0) value = static_cast<std::uint64_t>(*s);
    else { found = false; return 0; }
    if (value > std::numeric_limits<std::uint32_t>::max()) { found = false; return 0; }
    found = true; return static_cast<std::uint32_t>(value);
}
} // namespace

Qwen2Tokenizer::Qwen2Tokenizer(const GgufModelInfo& model) {
    const auto ti = model.metadata.find("tokenizer.ggml.tokens");
    if (ti == model.metadata.end() || !std::holds_alternative<std::vector<std::string>>(ti->second))
        throw std::runtime_error("GGUF is missing tokenizer.ggml.tokens");
    tokens_ = std::get<std::vector<std::string>>(ti->second);
    if (tokens_.empty() || tokens_.size() > std::numeric_limits<std::uint32_t>::max())
        throw std::runtime_error("Invalid tokenizer vocabulary size");
    token_ids_.reserve(tokens_.size());
    for (std::size_t i = 0; i < tokens_.size(); ++i) {
        if (!token_ids_.emplace(tokens_[i], static_cast<std::uint32_t>(i)).second)
            throw std::runtime_error("Duplicate token string in GGUF vocabulary");
        if (is_special(tokens_[i])) special_tokens_.push_back(tokens_[i]);
    }
    std::sort(special_tokens_.begin(), special_tokens_.end(),
              [](const auto& a, const auto& b) { return a.size() > b.size(); });
    const auto mi = model.metadata.find("tokenizer.ggml.merges");
    if (mi != model.metadata.end()) {
        if (!std::holds_alternative<std::vector<std::string>>(mi->second))
            throw std::runtime_error("tokenizer.ggml.merges must be a string array");
        const auto& merges = std::get<std::vector<std::string>>(mi->second);
        merge_ranks_.reserve(merges.size());
        for (std::size_t i = 0; i < merges.size(); ++i) {
            const auto split = merges[i].find(' ');
            if (split == std::string::npos || split == 0 || split + 1 >= merges[i].size()) continue;
            merge_ranks_.emplace(merge_key(merges[i].substr(0, split), merges[i].substr(split + 1)),
                                 static_cast<std::uint32_t>(std::min<std::size_t>(i, std::numeric_limits<std::uint32_t>::max())));
        }
    }
    bool found = false;
    eos_id_ = metadata_id(model, "tokenizer.ggml.eos_token_id", found);
    has_eos_ = found;
}

std::vector<std::uint32_t> Qwen2Tokenizer::encode(const std::string& text) const {
    std::vector<std::uint32_t> result;
    auto encode_piece = [&](const std::string& piece) {
      for (const auto& chunk : pretokenize(piece)) {
        std::vector<std::string> symbols;
        symbols.reserve(chunk.size());
        for (const unsigned char c : chunk) symbols.push_back(byte_symbol(c));
        while (symbols.size() > 1) {
            std::size_t best = symbols.size();
            std::uint32_t best_rank = std::numeric_limits<std::uint32_t>::max();
            for (std::size_t i = 0; i + 1 < symbols.size(); ++i) {
                const auto it = merge_ranks_.find(merge_key(symbols[i], symbols[i + 1]));
                if (it != merge_ranks_.end() && it->second < best_rank) {
                    best_rank = it->second; best = i;
                }
            }
            if (best == symbols.size()) break;
            symbols[best] += symbols[best + 1];
            symbols.erase(symbols.begin() + static_cast<std::ptrdiff_t>(best + 1));
        }
        for (const auto& symbol : symbols) {
            const auto it = token_ids_.find(symbol);
            if (it == token_ids_.end()) throw std::runtime_error("Tokenizer cannot encode a byte sequence in the model vocabulary");
            result.push_back(it->second);
        }
      }
    };

    std::size_t pos = 0;
    while (pos < text.size()) {
        std::size_t special_at = std::string::npos;
        std::string matched;
        for (const auto& special : special_tokens_) {
            const auto at = text.find(special, pos);
            if (at < special_at) { special_at = at; matched = special; }
        }
        if (special_at == std::string::npos) {
            encode_piece(text.substr(pos));
            break;
        }
        if (special_at > pos) encode_piece(text.substr(pos, special_at - pos));
        const auto it = token_ids_.find(matched);
        if (it == token_ids_.end()) throw std::runtime_error("Internal special-token lookup failure");
        result.push_back(it->second);
        pos = special_at + matched.size();
    }
    return result;
}

std::string Qwen2Tokenizer::decode(const std::vector<std::uint32_t>& ids, bool skip_special_tokens) const {
    std::string out;
    const auto& inverse = codepoint_to_byte();
    for (const auto id : ids) {
        if (id >= tokens_.size()) throw std::runtime_error("Token ID is outside the model vocabulary");
        const auto& token = tokens_[id];
        if (is_special(token)) {
            if (!skip_special_tokens) out += token;
            continue;
        }
        std::size_t i = 0;
        while (i < token.size()) {
            const auto before = i;
            const auto cp = read_cp(token, i);
            const auto it = inverse.find(cp);
            if (it != inverse.end()) out.push_back(static_cast<char>(it->second));
            else out.append(token, before, i - before);
        }
    }
    return out;
}
} // namespace brahma
