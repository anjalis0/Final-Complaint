const { Pool } = require('pg');

const pool = new Pool({
  user: 'postgres',
  host: 'localhost',
  database: 'postgres',
  password: 'anjali@12',
  port: 5432,
});

async function setupDatabase() {
  try {
    // Check if users table exists
    const tableCheck = await pool.query(`
      SELECT EXISTS (
        SELECT FROM information_schema.tables 
        WHERE table_schema = 'public' 
        AND table_name = 'users'
      );
    `);
    
    if (!tableCheck.rows[0].exists) {
      console.log('Creating users table...');
      await pool.query(`
        CREATE TABLE users (
          id SERIAL PRIMARY KEY,
          name VARCHAR(100) NOT NULL,
          student_id VARCHAR(8) UNIQUE NOT NULL,
          password VARCHAR(255) NOT NULL,
          role VARCHAR(20) DEFAULT 'student',
          created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
      `);
      console.log('Users table created successfully!');
    } else {
      console.log('Users table already exists.');

      // Ensure required columns exist (especially created_at used by /api/users)
      try {
        const columnCheck = await pool.query(`
          SELECT column_name 
          FROM information_schema.columns 
          WHERE table_name = 'users' AND column_name = 'created_at'
        `);

        if (columnCheck.rows.length === 0) {
          console.log('Adding missing created_at column to users table...');
          await pool.query(`
            ALTER TABLE users 
            ADD COLUMN created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
          `);
          console.log('created_at column added successfully.');
        } else {
          console.log('created_at column already exists.');
        }
      } catch (alterErr) {
        console.error('Error ensuring users table columns:', alterErr);
      }
    }
    
    // Show table structure
    const columns = await pool.query(`
      SELECT column_name, data_type, is_nullable 
      FROM information_schema.columns 
      WHERE table_name = 'users' 
      ORDER BY ordinal_position;
    `);
    
    console.log('\nUsers table structure:');
    columns.rows.forEach(col => {
      console.log(`- ${col.column_name}: ${col.data_type} (${col.is_nullable === 'YES' ? 'nullable' : 'not null'})`);
    });
    
  } catch (err) {
    console.error('Database setup error:', err);
  } finally {
    pool.end();
  }
}

setupDatabase();