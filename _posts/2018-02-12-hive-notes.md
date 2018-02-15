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

## Create Database
A database in Hive is a namespace or a collection of tables. The syntax for this statement is as follows:
```hive
CREATE DATABASE|SCHEMA [IF NOT EXISTS] <database name>
hive> CREATE DATABASE [IF NOT EXISTS] userdb;
SHOW DATABASES;
```
## Drop Database

```hive
DROP DATABASE StatementDROP (DATABASE|SCHEMA) [IF EXISTS] database_name 
[RESTRICT|CASCADE];
hive> DROP DATABASE IF EXISTS userdb;
```

The following query drops the database using CASCADE. It means dropping respective tables before dropping the database.

```hive
hive> DROP DATABASE IF EXISTS userdb CASCADE;
```
## Create Table
```hive
CREATE TABLE IF NOT EXISTS employee (eid int, name string,
salary float, destination string)
COMMENT 'Employee details'
ROW FORMAT DELIMITED
FIELDS TERMINATED BY '\t'
LINES TERMINATED BY '\n'
STORED AS TEXTFILE;
```
### Load Data
There are two ways to load data into hive. One is from local file system or from Hadoop file system. We can save the data in a text file and load them as follows.
```hive
1201  Gopal       45000    Technical manager
1202  Manisha     45000    Proof reader
1203  Masthanvali 40000    Technical writer
1204  Kiran       40000    Hr Admin
1205  Kranthi     30000    Op Admin
```

```hive
LOAD DATA LOCAL INPATH '/home/user/sample.txt'
OVERWRITE INTO TABLE employee;
```
## Alter Table
```hive
ALTER TABLE name RENAME TO new_name
ALTER TABLE name ADD COLUMNS (col_spec[, col_spec ...])
ALTER TABLE name DROP [COLUMN] column_name
ALTER TABLE name CHANGE column_name new_name new_type
ALTER TABLE name REPLACE COLUMNS (col_spec[, col_spec ...])
```

## Drop Table
## Partitioning
## Built-in Operators
## Built-in Functions
## Views and Indices

