// Script to migrate complaints data from main database to complaints database
const { Pool } = require('pg');

// Create connection pools
const mainPool = new Pool({
  user: 'postgres',
  host: 'localhost',
  database: 'postgres',
  password: 'anjali@12', // Use the same password as in server.js
  port: 5432,
});

const complaintsPool = new Pool({
  user: 'postgres',
  host: 'localhost',
  database: 'complaints_db',
  password: 'anjali@12', // Use the same password as in server.js
  port: 5432,
});

async function migrateComplaints() {
  try {
    console.log('Starting complaints migration...');
    
    // Get all complaints from main database
    const result = await mainPool.query('SELECT * FROM complaints');
    const complaints = result.rows;
    
    console.log(`Found ${complaints.length} complaints to migrate`);
    
    if (complaints.length === 0) {
      console.log('No complaints to migrate');
      return;
    }
    
    // Create complaints table in new database if it doesn't exist
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
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
      )
    `);
    
    // Create complaint_status_history table if it doesn't exist
    await complaintsPool.query(`
      CREATE TABLE IF NOT EXISTS complaint_status_history (
        id SERIAL PRIMARY KEY,
        complaint_id INTEGER NOT NULL,
        status VARCHAR(50) NOT NULL,
        admin_notes TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
      )
    `);
    
    // Insert each complaint into the new database
    for (const complaint of complaints) {
      // Handle null values for required fields
      const category = complaint.category || 'General';
      const status = complaint.status || 'pending';
      const priority = complaint.priority || 'medium';
      const sentiment = complaint.sentiment || 'neutral';
      
      await complaintsPool.query(`
        INSERT INTO complaints (
          id, user_id, title, description, category, status, priority, sentiment, admin_notes, created_at, updated_at
        ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11)
      `, [
        complaint.id,
        complaint.user_id,
        complaint.title,
        complaint.description,
        category,
        status,
        priority,
        sentiment,
        complaint.admin_notes,
        complaint.created_at || new Date(),
        complaint.updated_at || new Date()
      ]);
      
      console.log(`Migrated complaint ID: ${complaint.id}`);
    }
    
    // Get complaint status history if it exists
    try {
      const historyResult = await mainPool.query('SELECT * FROM complaint_status_history');
      const historyEntries = historyResult.rows;
      
      console.log(`Found ${historyEntries.length} history entries to migrate`);
      
      // Insert each history entry into the new database
      for (const entry of historyEntries) {
        await complaintsPool.query(`
          INSERT INTO complaint_status_history (
            id, complaint_id, status, admin_notes, created_at
          ) VALUES ($1, $2, $3, $4, $5)
        `, [
          entry.id,
          entry.complaint_id,
          entry.status,
          entry.admin_notes,
          entry.created_at
        ]);
        
        console.log(`Migrated history entry ID: ${entry.id}`);
      }
    } catch (err) {
      console.log('No complaint status history table found or error:', err.message);
    }
    
    console.log('Migration completed successfully!');
  } catch (err) {
    console.error('Migration error:', err);
  } finally {
    // Close the database connections
    mainPool.end();
    complaintsPool.end();
  }
}

// Run the migration
migrateComplaints();