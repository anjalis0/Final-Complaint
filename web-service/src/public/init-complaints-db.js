// Script to initialize the complaints database tables
const { Pool } = require('pg');

// Create connection pools
const mainPool = new Pool({
  user: 'postgres',
  host: 'localhost',
  database: 'postgres',
  password: 'anjali@12',
  port: 5432,
});

const complaintsPool = new Pool({
  user: 'postgres',
  host: 'localhost',
  database: 'complaints_db',
  password: 'anjali@12',
  port: 5432,
});

// Function to create the complaints database
async function createComplaintsDatabase() {
  try {
    // Check if database exists
    const dbCheckResult = await mainPool.query(
      "SELECT 1 FROM pg_database WHERE datname = 'complaints_db'"
    );
    
    // If database doesn't exist, create it
    if (dbCheckResult.rows.length === 0) {
      console.log('Creating complaints_db database...');
      await mainPool.query('CREATE DATABASE complaints_db');
      console.log('complaints_db database created successfully');
    } else {
      console.log('complaints_db database already exists');
    }
    
    return true;
  } catch (err) {
    console.error(`Error creating complaints database: ${err.message}`);
    return false;
  }
}

// Initialize complaints database tables
async function initializeComplaintsTables() {
  try {
    // Create complaints table if it doesn't exist
    await complaintsPool.query(`
      CREATE TABLE IF NOT EXISTS complaints (
        id SERIAL PRIMARY KEY,
        user_id INTEGER NOT NULL,
        title VARCHAR(255) NOT NULL,
        description TEXT NOT NULL,
        category VARCHAR(100) NOT NULL,
        status VARCHAR(50) NOT NULL DEFAULT 'pending',
        priority VARCHAR(50) NOT NULL DEFAULT 'medium',
        sentiment VARCHAR(50) DEFAULT 'neutral',
        admin_notes TEXT,
        attachment_path TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
      )
    `);
    console.log('Complaints table initialized');
    
    // Fix the sequence for the complaints table to prevent primary key violations
    await complaintsPool.query(`
      SELECT setval('complaints_id_seq', COALESCE((SELECT MAX(id) FROM complaints), 0) + 1, false);
    `);
    
    // Create complaint_status_history table if it doesn't exist
    await complaintsPool.query(`
      CREATE TABLE IF NOT EXISTS complaint_status_history (
        id SERIAL PRIMARY KEY,
        complaint_id INTEGER NOT NULL,
        status VARCHAR(50) NOT NULL,
        admin_notes TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
      )
    `);
    
    // Fix the sequence for the complaint_status_history table
    await complaintsPool.query(`
      SELECT setval('complaint_status_history_id_seq', COALESCE((SELECT MAX(id) FROM complaint_status_history), 0) + 1, false);
    `);
    
    console.log('Complaints database tables initialized successfully');
    return true;
  } catch (err) {
    console.error('Error initializing complaints database tables:', err);
    return false;
  }
}

// Run the initialization process
async function init() {
  try {
    // First create the database
    const dbCreated = await createComplaintsDatabase();
    if (!dbCreated) {
      throw new Error('Failed to create complaints database');
    }
    
    // Then initialize the tables
    const tablesInitialized = await initializeComplaintsTables();
    if (!tablesInitialized) {
      throw new Error('Failed to initialize complaints database tables');
    }
    
    console.log('Database initialization completed successfully');
    process.exit(0);
  } catch (err) {
    console.error('Initialization error:', err);
    process.exit(1);
  }
}

// Start the initialization
init();