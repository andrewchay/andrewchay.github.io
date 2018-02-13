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
### Literals
### NULL
### Complex Types 
```
