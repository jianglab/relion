#ifndef CRYOSPARC_JSON_H
#define CRYOSPARC_JSON_H

/* A small, dependency-free JSON reader, enough for CryoSPARC's job.json.
 *
 * RELION vendors npy.hpp and Eigen but has no JSON library, and job.json needs
 * only a read-only value tree: objects, arrays, strings, numbers, booleans and
 * null. That is a few hundred lines, so it is hand-rolled here rather than
 * vendoring a large single-header library for one caller.
 *
 * The parser is deliberately strict about structure and lenient about content:
 * a malformed file raises, but unknown keys are simply carried along, so a
 * CryoSPARC version that adds fields does not break the import.
 */

#include <string>
#include <vector>
#include <map>
#include <memory>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <stdexcept>

namespace cryosparc {

class JsonValue;
typedef std::shared_ptr<JsonValue> JsonPtr;

class JsonValue {
public:
	enum Type { NULLVAL, BOOL, NUMBER, STRING, ARRAY, OBJECT };

	Type type;
	bool bool_value;
	double number_value;
	std::string string_value;
	std::vector<JsonPtr> array_value;
	std::map<std::string, JsonPtr> object_value;

	JsonValue() : type(NULLVAL), bool_value(false), number_value(0.) {}

	bool isNull()   const { return type == NULLVAL; }
	bool isArray()  const { return type == ARRAY; }
	bool isObject() const { return type == OBJECT; }

	/// Member of an object, or an empty value when absent. Never returns null,
	/// so chains like get("a")->get("b")->asString() are safe on partial data.
	JsonPtr get(const std::string& key) const
	{
		if (type == OBJECT)
		{
			std::map<std::string, JsonPtr>::const_iterator it = object_value.find(key);
			if (it != object_value.end() && it->second) return it->second;
		}
		return emptyValue();
	}

	bool has(const std::string& key) const
	{
		return type == OBJECT && object_value.count(key) > 0;
	}

	size_t size() const
	{
		if (type == ARRAY)  return array_value.size();
		if (type == OBJECT) return object_value.size();
		return 0;
	}

	JsonPtr at(size_t i) const
	{
		if (type == ARRAY && i < array_value.size() && array_value[i]) return array_value[i];
		return emptyValue();
	}

	std::string asString(const std::string& fallback = "") const
	{
		if (type == STRING) return string_value;
		return fallback;
	}

	double asDouble(double fallback = 0.) const
	{
		if (type == NUMBER) return number_value;
		if (type == BOOL)   return bool_value ? 1. : 0.;
		return fallback;
	}

	long asLong(long fallback = 0) const
	{
		if (type == NUMBER) return (long)llround(number_value);
		if (type == BOOL)   return bool_value ? 1 : 0;
		return fallback;
	}

	bool asBool(bool fallback = false) const
	{
		if (type == BOOL)   return bool_value;
		if (type == NUMBER) return number_value != 0.;
		return fallback;
	}

	/// Array of strings, skipping anything that is not a string
	std::vector<std::string> asStringVector() const
	{
		std::vector<std::string> out;
		if (type != ARRAY) return out;
		for (size_t i = 0; i < array_value.size(); i++)
			if (array_value[i] && array_value[i]->type == STRING)
				out.push_back(array_value[i]->string_value);
		return out;
	}

private:
	static JsonPtr emptyValue()
	{
		static JsonPtr empty(new JsonValue());
		return empty;
	}
};

namespace json_detail {

struct Parser {
	const std::string& s;
	size_t i;

	explicit Parser(const std::string& text) : s(text), i(0) {}

	[[noreturn]] void fail(const std::string& what) const
	{
		throw std::runtime_error("JSON parse error at offset " +
		                         std::to_string(i) + ": " + what);
	}

	void skipSpace()
	{
		while (i < s.size() && (s[i] == ' ' || s[i] == '\t' || s[i] == '\n' || s[i] == '\r'))
			i++;
	}

	bool literal(const char* lit)
	{
		const size_t n = strlen(lit);
		if (s.compare(i, n, lit) == 0) { i += n; return true; }
		return false;
	}

	std::string parseString()
	{
		if (i >= s.size() || s[i] != '"') fail("expected a string");
		i++;
		std::string out;
		while (i < s.size() && s[i] != '"')
		{
			if (s[i] == '\\')
			{
				i++;
				if (i >= s.size()) fail("unterminated escape");
				switch (s[i])
				{
				case '"':  out += '"';  break;
				case '\\': out += '\\'; break;
				case '/':  out += '/';  break;
				case 'b':  out += '\b'; break;
				case 'f':  out += '\f'; break;
				case 'n':  out += '\n'; break;
				case 'r':  out += '\r'; break;
				case 't':  out += '\t'; break;
				case 'u':
				{
					// Keep it simple: decode the BMP code point as UTF-8.
					// CryoSPARC metadata is effectively ASCII, so this is a
					// fallback rather than a hot path.
					if (i + 4 >= s.size()) fail("truncated \\u escape");
					const unsigned cp = (unsigned)strtoul(s.substr(i + 1, 4).c_str(), NULL, 16);
					i += 4;
					if (cp < 0x80) out += (char)cp;
					else if (cp < 0x800)
					{
						out += (char)(0xC0 | (cp >> 6));
						out += (char)(0x80 | (cp & 0x3F));
					}
					else
					{
						out += (char)(0xE0 | (cp >> 12));
						out += (char)(0x80 | ((cp >> 6) & 0x3F));
						out += (char)(0x80 | (cp & 0x3F));
					}
					break;
				}
				default: fail("unknown escape");
				}
				i++;
			}
			else out += s[i++];
		}
		if (i >= s.size()) fail("unterminated string");
		i++; // closing quote
		return out;
	}

	JsonPtr parseValue()
	{
		skipSpace();
		if (i >= s.size()) fail("unexpected end of input");

		JsonPtr v(new JsonValue());

		switch (s[i])
		{
		case '{':
		{
			i++;
			v->type = JsonValue::OBJECT;
			skipSpace();
			if (i < s.size() && s[i] == '}') { i++; return v; }
			while (true)
			{
				skipSpace();
				const std::string key = parseString();
				skipSpace();
				if (i >= s.size() || s[i] != ':') fail("expected ':'");
				i++;
				v->object_value[key] = parseValue();
				skipSpace();
				if (i < s.size() && s[i] == ',') { i++; continue; }
				if (i < s.size() && s[i] == '}') { i++; break; }
				fail("expected ',' or '}'");
			}
			return v;
		}
		case '[':
		{
			i++;
			v->type = JsonValue::ARRAY;
			skipSpace();
			if (i < s.size() && s[i] == ']') { i++; return v; }
			while (true)
			{
				v->array_value.push_back(parseValue());
				skipSpace();
				if (i < s.size() && s[i] == ',') { i++; continue; }
				if (i < s.size() && s[i] == ']') { i++; break; }
				fail("expected ',' or ']'");
			}
			return v;
		}
		case '"':
			v->type = JsonValue::STRING;
			v->string_value = parseString();
			return v;
		default:
			if (literal("true"))  { v->type = JsonValue::BOOL; v->bool_value = true;  return v; }
			if (literal("false")) { v->type = JsonValue::BOOL; v->bool_value = false; return v; }
			if (literal("null"))  { v->type = JsonValue::NULLVAL; return v; }
			// NaN / Infinity are not valid JSON but Python's json.dump emits them,
			// and CryoSPARC writes job.json with Python.
			if (literal("NaN"))       { v->type = JsonValue::NUMBER; v->number_value = NAN; return v; }
			if (literal("Infinity"))  { v->type = JsonValue::NUMBER; v->number_value = INFINITY; return v; }
			if (literal("-Infinity")) { v->type = JsonValue::NUMBER; v->number_value = -INFINITY; return v; }
			{
				char* end = NULL;
				const double d = strtod(s.c_str() + i, &end);
				if (end == s.c_str() + i) fail("expected a value");
				i = (size_t)(end - s.c_str());
				v->type = JsonValue::NUMBER;
				v->number_value = d;
				return v;
			}
		}
	}
};

} // namespace json_detail

/// Parse a JSON document. Throws std::runtime_error on malformed input.
inline JsonPtr jsonParse(const std::string& text)
{
	json_detail::Parser p(text);
	JsonPtr v = p.parseValue();
	p.skipSpace();
	return v;
}

/// Parse a JSON file. Throws std::runtime_error if it cannot be read or parsed.
inline JsonPtr jsonParseFile(const std::string& filename)
{
	FILE* f = fopen(filename.c_str(), "rb");
	if (f == NULL)
		throw std::runtime_error("cannot open JSON file: " + filename);

	std::string text;
	char buf[65536];
	size_t n;
	while ((n = fread(buf, 1, sizeof(buf), f)) > 0) text.append(buf, n);
	fclose(f);

	try { return jsonParse(text); }
	catch (const std::runtime_error& e)
	{
		throw std::runtime_error(std::string(e.what()) + " (in " + filename + ")");
	}
}

} // namespace cryosparc

#endif // CRYOSPARC_JSON_H
