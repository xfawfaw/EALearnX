//+------------------------------------------------------------------+
//|                                                     JsonLite.mqh |
//|                     Lightweight flat/nested JSON helper for MQL5 |
//+------------------------------------------------------------------+
#property copyright "EA-LearningX"
#property version   "1.00"

#ifndef __JSON_LITE_MQH__
#define __JSON_LITE_MQH__

//+------------------------------------------------------------------+
//| Escape a string for safe JSON serialization                      |
//+------------------------------------------------------------------+
string JsonEscape(const string text)
  {
   string out = "";
   int len = StringLen(text);
   for(int i = 0; i < len; i++)
     {
      ushort ch = StringGetCharacter(text, i);
      switch(ch)
        {
         case '\"': out += "\\\""; break;
         case '\\': out += "\\\\"; break;
         case 8:    out += "\\b";  break;
         case 12:   out += "\\f";  break;
         case '\n': out += "\\n";  break;
         case '\r': out += "\\r";  break;
         case '\t': out += "\\t";  break;
         default:
            if(ch < 32)
               out += " ";
            else
               out += ShortToString(ch);
            break;
        }
     }
   return out;
  }

//+------------------------------------------------------------------+
//| Find the value token start index for a given top-level key       |
//+------------------------------------------------------------------+
int JsonFindKeyValueStart(const string json, const string key)
  {
   string pattern = "\"" + key + "\"";
   int pos = 0;
   int len = StringLen(json);

   while(pos < len)
     {
      int found = StringFind(json, pattern, pos);
      if(found < 0)
         return -1;

      int after = found + StringLen(pattern);
      while(after < len)
        {
         ushort c = StringGetCharacter(json, after);
         if(c == ' ' || c == '\t' || c == '\r' || c == '\n')
            after++;
         else
            break;
        }

      if(after < len && StringGetCharacter(json, after) == ':')
        {
         after++;
         while(after < len)
           {
            ushort c = StringGetCharacter(json, after);
            if(c == ' ' || c == '\t' || c == '\r' || c == '\n')
               after++;
            else
               break;
           }
         return after;
        }
      pos = found + StringLen(pattern);
     }
   return -1;
  }

//+------------------------------------------------------------------+
//| Extract string value by key                                      |
//+------------------------------------------------------------------+
bool JsonGetString(const string json, const string key, string &value, const string default_val = "")
  {
   value = default_val;
   int start = JsonFindKeyValueStart(json, key);
   if(start < 0)
      return false;

   int len = StringLen(json);
   if(start >= len)
      return false;

   if(StringGetCharacter(json, start) != '\"')
      return false;

   string result = "";
   int i = start + 1;
   while(i < len)
     {
      ushort ch = StringGetCharacter(json, i);
      if(ch == '\\' && i + 1 < len)
        {
         ushort next = StringGetCharacter(json, i + 1);
         switch(next)
           {
            case '\"': result += "\""; break;
            case '\\': result += "\\"; break;
            case '/':  result += "/";  break;
            case 'n':  result += "\n"; break;
            case 'r':  result += "\r"; break;
            case 't':  result += "\t"; break;
            default:   result += ShortToString(next); break;
           }
         i += 2;
         continue;
        }
      if(ch == '\"')
        {
         value = result;
         return true;
        }
      result += ShortToString(ch);
      i++;
     }
   return false;
  }

//+------------------------------------------------------------------+
//| Extract raw primitive token (number / bool / null)               |
//+------------------------------------------------------------------+
bool JsonGetRawToken(const string json, const string key, string &token)
  {
   token = "";
   int start = JsonFindKeyValueStart(json, key);
   if(start < 0)
      return false;

   int len = StringLen(json);
   int i = start;
   while(i < len)
     {
      ushort ch = StringGetCharacter(json, i);
      if(ch == ',' || ch == '}' || ch == ']' || ch == ' ' || ch == '\t' || ch == '\r' || ch == '\n')
         break;
      i++;
     }
   if(i <= start)
      return false;

   token = StringSubstr(json, start, i - start);
   return true;
  }

//+------------------------------------------------------------------+
//| Extract double value by key                                      |
//+------------------------------------------------------------------+
bool JsonGetDouble(const string json, const string key, double &value, const double default_val = 0.0)
  {
   value = default_val;
   string tok;
   if(!JsonGetRawToken(json, key, tok))
      return false;
   if(tok == "null")
      return false;
   value = StringToDouble(tok);
   return true;
  }

//+------------------------------------------------------------------+
//| Extract integer (long) value by key                              |
//+------------------------------------------------------------------+
bool JsonGetLong(const string json, const string key, long &value, const long default_val = 0)
  {
   value = default_val;
   string tok;
   if(!JsonGetRawToken(json, key, tok))
      return false;
   if(tok == "null")
      return false;
   value = StringToInteger(tok);
   return true;
  }

//+------------------------------------------------------------------+
//| Extract boolean value by key                                     |
//+------------------------------------------------------------------+
bool JsonGetBool(const string json, const string key, bool &value, const bool default_val = false)
  {
   value = default_val;
   string tok;
   if(!JsonGetRawToken(json, key, tok))
      return false;
   if(tok == "true")
     {
      value = true;
      return true;
     }
   if(tok == "false")
     {
      value = false;
      return true;
     }
   return false;
  }

//+------------------------------------------------------------------+
//| Read UTF-8 text file from FILE_COMMON                            |
//+------------------------------------------------------------------+
bool ReadCommonUtf8File(const string rel_path, string &content)
  {
   content = "";
   if(!FileIsExist(rel_path, FILE_COMMON))
      return false;

   int handle = FileOpen(rel_path, FILE_READ | FILE_BIN | FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_COMMON);
   if(handle == INVALID_HANDLE)
      return false;

   ulong size = FileSize(handle);
   if(size == 0)
     {
      FileClose(handle);
      return false;
     }

   uchar bytes[];
   ArrayResize(bytes, (int)size);
   uint read_bytes = FileReadArray(handle, bytes, 0, (int)size);
   FileClose(handle);

   if(read_bytes == 0)
      return false;

   content = CharArrayToString(bytes, 0, (int)read_bytes, CP_UTF8);
   return (StringLen(content) > 0);
  }

//+------------------------------------------------------------------+
//| Atomically write UTF-8 text file to FILE_COMMON                  |
//+------------------------------------------------------------------+
bool WriteCommonUtf8File(const string rel_path, const string content)
  {
   string tmp_path = rel_path + ".tmp";
   int handle = FileOpen(tmp_path, FILE_WRITE | FILE_BIN | FILE_COMMON);
   if(handle == INVALID_HANDLE)
      return false;

   uchar bytes[];
   int count = StringToCharArray(content, bytes, 0, WHOLE_ARRAY, CP_UTF8);
   if(count > 1)
     {
      // StringToCharArray includes trailing '\0'; strip it for clean UTF-8 files
      FileWriteArray(handle, bytes, 0, count - 1);
     }
   FileClose(handle);

   if(!FileMove(tmp_path, FILE_COMMON, rel_path, FILE_COMMON | FILE_REWRITE))
     {
      FileDelete(tmp_path, FILE_COMMON);
      return false;
     }
   return true;
  }

#endif // __JSON_LITE_MQH__
//+------------------------------------------------------------------+
