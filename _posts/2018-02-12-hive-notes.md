# This is the notes on Hadoop Hive.
## Hive Data Types
### Column Types
1. **Integer Types**.
   | Type  | PostFix  |
   |---|---|
   | INT  | -  |
   | BIGINT  |  L |
   | SMALLINT  | S  |
   | TINYINT  | Y  |
2. **String Types**.
VARCHAR (1 to 65355) or CHAR (255)
3. **Timestamp**.
“YYYY-MM-DD HH:MM:SS.fffffffff”
4. **Dates**. {{YYYY-MM-DD}}
5. **Decimals**. decimal(10,0)
6. **Union Types**.
```hive 
UNIONTYPE<int, double, array<string>, struct<a:int,b:string>>

{0:1} 
{1:2.0} 
{2:["three","four"]} 
{3:{"a":5,"b":"five"}} 
{2:["six","seven"]} 
{3:{"a":8,"b":"eight"}} 
{0:9} 
{1:10.0}
```
### Literals
1. **Floating Point Types**.
Floating point types are nothing but numbers with decimal points. Generally, this type of data is composed of DOUBLE data type.
2. **Decimal Type**.
Decimal type data is nothing but floating point value with higher range than DOUBLE data type. The range of decimal type is approximately -10-308 to 10308.
### NULL
Missing value is saved as NULL.
### Complex Types 
The Hive complex data types are as follows:
1. **Arrays**.
Arrays in Hive are used the same way they are used in Java. Syntax: ARRAY<data_type>
2. **Maps**.
Maps in Hive are similar to Java Maps. Syntax: MAP<primitive_type, data_type>
3. **Structs**.
Structs in Hive is similar to using complex data with comment. Syntax: STRUCT<col_name : data_type [COMMENT col_comment], ...>