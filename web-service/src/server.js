const express = require('express');
const cors = require('cors');
const bcrypt = require('bcrypt');
const { Pool } = require('pg');
const axios = require('axios');
const multer = require('multer');
const path = require('path');
const fs = require('fs');
// Simple logger replacement with environment-based logging
const NODE_ENV = process.env.NODE_ENV || 'development';
const logger = {
  info: (msg) => console.log(`[INFO] ${msg}`),
  error: (msg, err) => console.error(`[ERROR] ${msg}`, err || ''),
  warn: (msg) => console.warn(`[WARN] ${msg}`),
  debug: (msg) => NODE_ENV === 'development' ? console.log(`[DEBUG] ${msg}`) : null
};
const app = express();

app.use(cors({
  origin: '*',
  credentials: true,
  methods: ['GET', 'POST', 'PUT', 'DELETE', 'OPTIONS'],
  allowedHeaders: ['Content-Type', 'Authorization']
}));
app.use(express.json());
app.use(express.urlencoded({ extended: true })); // Add this line to parse URL-encoded form data
app.use(express.static(path.join(__dirname, 'public'))); // Serve static files from public folder

app.use('/uploads', express.static(path.join(__dirname, 'public/uploads'))); // Explicitly serve files from uploads directory

// Serve HTML files directly
app.get('/', (req, res) => {
  res.sendFile(path.join(__dirname, 'git push --set-upstream https://github.com/anjalis0/Final-Complaint feature-complain-system/web_server/src/index.html'));
});

app.get('/login.html', (req, res) => {
  res.sendFile(path.join(__dirname, 'public/login.html'));
});

app.get('/signup.html', (req, res) => {
  res.sendFile(path.join(__dirname, 'public/signup.html'));
});

// Create uploads directory if it doesn't exist
const uploadsDir = path.join(__dirname, 'public/uploads');
if (!fs.existsSync(uploadsDir)) {
  fs.mkdirSync(uploadsDir, { recursive: true });
}

// Configure multer storage
const storage = multer.diskStorage({
  destination: function (req, file, cb) {
    cb(null, uploadsDir);
  },
  filename: function (req, file, cb) {
    const uniqueSuffix = Date.now() + '-' + Math.round(Math.random() * 1E9);
    cb(null, uniqueSuffix + path.extname(file.originalname));
  }
});

const upload = multer({ storage: storage });

// Main database pool for users and other data
const mainPool = new Pool({
  user: 'postgres',         // e.g., 'postgres'
  host: 'localhost',
  database: 'postgres',      // your database name
  password: 'anjali@12', // your password
  port: 5432,
});

// We'll create the complaints_db database first, then connect to it
let complaintsPool = null;

// Function to create the complaints database
async function createComplaintsDatabase() {
  try {
    // Check if database exists
    const dbCheckResult = await mainPool.query(
      "SELECT 1 FROM pg_database WHERE datname = 'complaints_db'"
    );
    
    // If database doesn't exist, create it
    if (dbCheckResult.rows.length === 0) {
      logger.info('Creating complaints_db database...');
      await mainPool.query('CREATE DATABASE complaints_db');
      logger.info('complaints_db database created successfully');
    } else {
      logger.info('complaints_db database already exists');
    }
    
    // Now create the connection pool to the complaints database
    complaintsPool = new Pool({
      user: 'postgres',         // e.g., 'postgres'
      host: 'localhost',
      database: 'complaints_db',      // separate database for complaints
      password: 'anjali@12', // your password
      port: 5432,
    });
    
    return true;
  } catch (err) {
    logger.error(`Error creating complaints database: ${err.message}`);
    return false;
  }
}

// Initialize complaints database if it doesn't exist
async function initializeComplaintsDB() {
  try {
    // First check if the database exists
    const checkDbResult = await mainPool.query(
      "SELECT 1 FROM pg_database WHERE datname = 'complaints_db'"
    );
    
    // If database doesn't exist, create it
    if (checkDbResult.rows.length === 0) {
      logger.info('Creating complaints_db database...');
      await mainPool.query('CREATE DATABASE complaints_db');
      
      // Connect to the new database and create tables
      complaintsPool = new Pool({
        user: 'postgres',
        host: '0.0.0.0',
        database: 'complaints_db',
        password: 'anjali@12',
        port: 5432,
      });
      
      // Create complaints table
      await complaintsPool.query(`
        CREATE TABLE IF NOT EXISTS complaints (
          id SERIAL PRIMARY KEY,
          title VARCHAR(255) NOT NULL,
          description TEXT,
          category VARCHAR(100),
          user_id INTEGER NOT NULL,
          status VARCHAR(50) DEFAULT 'pending',
          priority VARCHAR(50),
          sentiment VARCHAR(50),
          admin_notes TEXT,
          attachment_path TEXT,
          created_at TIMESTAMP DEFAULT NOW(),
          updated_at TIMESTAMP DEFAULT NOW()
        )
      `);
      
      // Create complaint status history table
      await complaintsPool.query(`
        CREATE TABLE IF NOT EXISTS complaint_status_history (
          id SERIAL PRIMARY KEY,
          complaint_id INTEGER REFERENCES complaints(id) ON DELETE CASCADE,
          status VARCHAR(50) NOT NULL,
          admin_notes TEXT,
          created_at TIMESTAMP DEFAULT NOW()
        )
      `);
      logger.info('Complaints database initialized successfully');
    } else {
      logger.info('Complaints database already exists');
      
      // Connect to the existing database and add attachment_path column if it doesn't exist
      try {
        // Check if attachment_path column exists
        const columnCheckResult = await complaintsPool.query(`
          SELECT column_name 
          FROM information_schema.columns 
          WHERE table_name='complaints' AND column_name='attachment_path'
        `);
        
        // If column doesn't exist, add it
        if (columnCheckResult.rows.length === 0) {
          logger.info('Adding attachment_path column to complaints table...');
          await complaintsPool.query(`
            ALTER TABLE complaints 
            ADD COLUMN IF NOT EXISTS attachment_path TEXT
          `);
          logger.info('attachment_path column added successfully');
        } else {
          logger.debug('attachment_path column already exists');
        }
      } catch (columnErr) {
        logger.error(`Error checking/adding attachment_path column: ${columnErr.message}`);
      }
    }
  } catch (error) {
    logger.error(`Error initializing complaints database: ${error.message}`);
    return false;
  }
  return true;
}

// Initialize the complaints database
initializeComplaintsDB();

// Signup endpoint
app.post('/api/signup', async (req, res) => {
  logger.info('Signup attempt received:', { student_id: req.body.student_id });
  const { name, student_id, password, role = 'student' } = req.body;
  
  // Validate student_id format (college specific pattern)
  if (!student_id || !/^2023\d{4}$/.test(student_id)) {
    logger.warn('Invalid student ID format:', student_id);
    return res.status(400).json({ success: false, message: 'Invalid Student ID format. Must be 2023 followed by 4 digits.' });
  }

  // Validate role
  if (!['student', 'teacher', 'admin'].includes(role)) {
    logger.warn('Invalid role:', role);
    return res.status(400).json({ success: false, message: 'Invalid role. Must be either student, teacher, or admin.' });
  }
  
  try {
    // First check if the users table exists
    try {
      const tableCheck = await mainPool.query(`
        SELECT EXISTS (
          SELECT FROM information_schema.tables 
          WHERE table_schema = 'public' 
          AND table_name = 'users'
        );
      `);
      
      if (!tableCheck.rows[0].exists) {
        logger.error('Users table does not exist');
        return res.status(500).json({ success: false, message: 'Database setup incomplete. Please run setup-db.js first.' });
      }
    } catch (tableErr) {
      logger.error('Error checking users table:', tableErr);
      return res.status(500).json({ success: false, message: 'Database connection error.' });
    }
    
    const hash = await bcrypt.hash(password, 10);
    // Check if admin already exists when role is admin
    if (role === 'admin') {
      const adminCheck = await mainPool.query('SELECT * FROM users WHERE role = $1', ['admin']);
      if (adminCheck.rows.length > 0) {
        logger.warn('Admin account already exists');
        return res.json({ success: false, message: 'Admin account already exists.' });
      }
    }

    await mainPool.query(
      'INSERT INTO users (name, student_id, password, role) VALUES ($1, $2, $3, $4)',
      [name, student_id, hash, role]
    );
    logger.info('User created successfully:', { student_id, role });
    res.json({ success: true, message: 'Signup successful!' });
  } catch (err) {
    if (err.code === '23505') { // unique_violation
      logger.warn('Duplicate student ID:', student_id);
      return res.status(409).json({ success: false, message: 'Student ID already registered.' });
    } else {
      logger.error('Signup error:', err);
      return res.status(500).json({ success: false, message: 'Signup failed: ' + err.message });
    }
  }
});

// Login endpoint
app.post('/api/login', async (req, res) => {
  logger.info('Login attempt received:', { student_id: req.body.student_id });
  const { student_id, password } = req.body;
  
  // Validate student_id format
  if (!student_id || !/^2023\d{4}$/.test(student_id)) {
    logger.warn('Invalid student ID format:', student_id);
    return res.status(400).json({ success: false, message: 'Invalid Student ID format. Must be 2023 followed by 4 digits.' });
  }
  
  try {
    logger.info('Querying database for student_id:', student_id);
    
    // First check if the users table exists
    try {
      const tableCheck = await mainPool.query(`
        SELECT EXISTS (
          SELECT FROM information_schema.tables 
          WHERE table_schema = 'public' 
          AND table_name = 'users'
        );
      `);
      
      if (!tableCheck.rows[0].exists) {
        logger.error('Users table does not exist');
        return res.status(500).json({ success: false, message: 'Database setup incomplete. Please run setup-db.js first.' });
      }
    } catch (tableErr) {
      logger.error('Error checking users table:', tableErr);
      return res.status(500).json({ success: false, message: 'Database connection error.' });
    }
    
    const result = await mainPool.query('SELECT * FROM users WHERE student_id=$1', [student_id]);
    logger.info('Database query result:', { found: result.rows.length > 0 });
    
    const user = result.rows[0];
    if (user && await bcrypt.compare(password, user.password)) {
      logger.info('Password match successful for user:', { student_id: user.student_id, role: user.role });
      res.json({ 
        success: true, 
        name: user.name, 
        student_id: user.student_id,
        role: user.role || 'student' // Default to student if role not set
      });
    } else {
      logger.warn('Login failed: Invalid credentials');
      return res.status(401).json({ success: false, message: 'Invalid credentials' });
    }
  } catch (err) {
    logger.error('Login error:', err);
    res.status(500).json({ success: false, message: 'Login failed: ' + err.message });
  }
});

// Multer, path, and fs are already imported at the top of the file
// Upload configuration is already set up at the top of the file

// Submit complaint endpoint with ML analysis and file upload
app.post('/api/submit-complaint', upload.single('attachment'), async (req, res) => {
  logger.info('Received form data:', req.body);
  logger.info('Received file:', req.file);
  
  const { title, description, category, user_id, student_id } = req.body;
  
  // Use student_id if provided, otherwise use user_id
  const userIdentifier = student_id || user_id;
  
  // Validate required fields
  if (!title || !description || !userIdentifier) {
    return res.status(400).json({ success: false, message: 'Missing required fields: ' + 
      (!title ? 'title ' : '') + 
      (!description ? 'description ' : '') + 
      (!userIdentifier ? 'user_id/student_id' : '') });
  }
  // Note: attachment and category fields are optional
  
  // Set default category if not provided
  const finalCategory = category || 'General';
  
  // First, get the actual user ID from the database using student_id
  let userId;
  try {
    logger.info(`Looking up user with student_id: ${userIdentifier}`);
    // Use mainPool instead of complaintsPool since users are in the main database
    const userResult = await mainPool.query('SELECT id FROM users WHERE student_id=$1', [userIdentifier]);
    logger.info(`User lookup result: ${JSON.stringify(userResult.rows)}`);
    if (userResult.rows.length === 0) {
      return res.status(404).json({ success: false, message: 'User not found' });
    }
    userId = userResult.rows[0].id;
  } catch (userErr) {
    logger.error('Error finding user:', userErr);
    return res.status(500).json({ success: false, message: `Failed to find user: ${userErr.message}` });
  }
  
  try {
    // First, analyze the complaint using ML service
    let sentiment = 'neutral';
    let priority = 'medium';
    
    try {
      const mlResponse = await axios.post('http://localhost:8000/predict', {
        text: `${title} ${description}`
      }, { timeout: 5000 });
      console.log('ML Response:', mlResponse.data);
      if (mlResponse.data.sentiment) {
        sentiment = mlResponse.data.sentiment.prediction;
      }
      if (mlResponse.data.priority) {
        priority = mlResponse.data.priority.prediction;
      }
    } catch (mlError) {
      logger.warn('ML service unavailable, using default values:', mlError.message);
    }
    
    // Get file attachment path if uploaded
    let attachment_path = null;
    if (req.file) {
      // Store the path relative to the server root
      attachment_path = '/uploads/' + req.file.filename;
      logger.info('Attachment path saved to database:', attachment_path);
    }
    
    // Insert complaint into database
    const result = await complaintsPool.query(
      'INSERT INTO complaints (user_id, title, description, category, sentiment, priority, status, attachment_path, created_at, updated_at) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, NOW(), NOW()) RETURNING *',
      [userId, title, description, finalCategory, sentiment, priority, 'pending', attachment_path]
    );
    
    logger.info('Complaint inserted successfully:', result.rows[0]);
    
    // Insert initial status history
    const historyResult = await complaintsPool.query(
      'INSERT INTO complaint_status_history (complaint_id, status) VALUES ($1, $2) RETURNING id',
      [result.rows[0].id, 'pending']
    );
    
    logger.info('Status history entry created:', historyResult.rows[0]);
    
    
    // Create a response object with the full URL for the attachment if it exists
    const responseObj = { 
      success: true, 
      message: 'Complaint submitted successfully!',
      complaint: result.rows[0],
      complaint_id: result.rows[0].id,
      analysis: {
        sentiment,
        priority
      }
    };
    
    // Add attachment information if a file was uploaded
    if (attachment_path) {
      responseObj.attachment_path = attachment_path;
      // Also include the full URL in the response
      const serverUrl = req.protocol + '://' + req.get('host');
      responseObj.attachment_url = serverUrl + attachment_path;
    }
    
    res.json(responseObj);
  } catch (err) {
    logger.error('Error submitting complaint:', err);
    res.status(500).json({ success: false, message: `Failed to submit complaint: ${err.message}` });
    return;
  }
});

// Get complaints endpoint
app.get('/api/complaints', async (req, res) => {
  const { user_id, role, since } = req.query;
  
  try {
    // First, get all complaints from the complaints database
    let complaintsQuery;
    let complaintsParams = [];
    let paramIndex = 1;
    
    // Base query for complaints
    let baseQuery = 'SELECT * FROM complaints c';
    
    // Add conditions
    let conditions = [];
    
    // Filter by user_id for non-admin users
    if (role !== 'admin' && user_id) {
      conditions.push(`c.user_id = $${paramIndex++}`);
      complaintsParams.push(user_id);
    }
    
    // Filter by timestamp if 'since' parameter is provided
    if (since) {
      console.log('Filtering complaints since:', since);
      conditions.push(`c.created_at > $${paramIndex++}::timestamp`);
      complaintsParams.push(since);
    }
    
    // Add WHERE clause if there are conditions
    if (conditions.length > 0) {
      baseQuery += ' WHERE ' + conditions.join(' AND ');
    }
    
    // Add ordering by priority (high, medium, low) and then by creation date
    complaintsQuery = baseQuery + ` ORDER BY 
      CASE 
        WHEN c.priority = 'high' THEN 1 
        WHEN c.priority = 'medium' THEN 2 
        WHEN c.priority = 'low' THEN 3 
        ELSE 4 
      END, 
      c.created_at DESC`;
    
    logger.debug(`Executing complaints query: ${complaintsQuery}`);
    logger.debug(`Query parameters: ${JSON.stringify(complaintsParams)}`);
    
    const complaintsResult = await complaintsPool.query(complaintsQuery, complaintsParams);
    const complaints = complaintsResult.rows;
    logger.info(`Found ${complaints.length} complaints`);
    
    // If we have complaints, get the user information for each complaint
    if (complaints.length > 0) {
      // Extract unique user IDs from complaints
      const userIds = [...new Set(complaints.map(c => c.user_id))];
      
      // Get user information for these IDs
      const userQuery = 'SELECT id, name, student_id FROM users WHERE id = ANY($1)';
      const userResult = await mainPool.query(userQuery, [userIds]);
      const users = userResult.rows;
      
      // Create a map of user_id to user info for quick lookup
      const userMap = {};
      users.forEach(user => {
        userMap[user.id] = user;
      });
      
      // Get server URL for attachment paths
      const serverUrl = req.protocol + '://' + req.get('host');
      
      // Combine complaint data with user data
      const completeComplaints = complaints.map(complaint => {
        const user = userMap[complaint.user_id] || { name: 'Unknown', student_id: 'Unknown' };
        
        // Create a complaint object with user data
        const completeComplaint = {
          ...complaint,
          user_name: user.name,
          student_id: user.student_id
        };
        
        // Add attachment URL if attachment path exists
        if (complaint.attachment_path) {
          completeComplaint.attachment_url = serverUrl + complaint.attachment_path;
        }
        
        return completeComplaint;
      });
      
      // Log the complaints with user information for debugging
      logger.debug(`Complaints with user info: ${JSON.stringify(completeComplaints.map(c => ({ 
        id: c.id, 
        title: c.title, 
        user_id: c.user_id, 
        user_name: c.user_name, 
        student_id: c.student_id,
        status: c.status,
        has_attachment: !!c.attachment_path
      })))}`);
      
      
      // Return the complete complaints with user data
      res.json({ success: true, complaints: completeComplaints });
    } else {
      // No complaints found
      res.json({ success: true, complaints: [] });
    }
  } catch (err) {
    logger.error(`Error fetching complaints: ${err.message}`);
    res.status(500).json({ success: false, message: 'Failed to fetch complaints.' });
  }
});

// Get a specific complaint endpoint (admin only)
app.get('/api/complaints/:id', async (req, res) => {
  const { id } = req.params;
  
  try {
    // Get the complaint from complaints database
    const complaintResult = await complaintsPool.query(
      'SELECT * FROM complaints WHERE id = $1',
      [id]
    );
    
    if (complaintResult.rows.length === 0) {
      return res.status(404).json({ success: false, message: 'Complaint not found' });
    }
    
    const complaint = complaintResult.rows[0];
    
    // Get status history if available
    let statusHistory = [];
    try {
      const historyResult = await complaintsPool.query(
        'SELECT status, admin_notes as note, timestamp FROM complaint_status_history WHERE complaint_id = $1 ORDER BY timestamp ASC',
        [id]
      );
      statusHistory = historyResult.rows;
      logger.debug(`Retrieved ${statusHistory.length} status history entries for complaint ${id}: ${JSON.stringify(statusHistory)}`);
      
    } catch (historyErr) {
      logger.error(`Status history not available: ${historyErr.message}`);
    }
    
    // Add status history to the complaint object
    complaint.status_history = statusHistory;
    
    // Add attachment URL if attachment path exists
    if (complaint.attachment_path) {
      const serverUrl = req.protocol + '://' + req.get('host');
      complaint.attachment_url = serverUrl + (complaint.attachment_path.startsWith('/') ? '' : '/') + complaint.attachment_path;
    }
    
    res.json({ success: true, complaint });
  } catch (err) {
    logger.error(`Error retrieving complaint: ${err.message}`);
    res.status(500).json({ success: false, message: 'Failed to retrieve complaint' });
  }
});

// Update complaint status endpoint (admin only)
app.put('/api/complaints/:id/status', async (req, res) => {
  const { id } = req.params;
  const { status, admin_notes } = req.body;
  
  try {
    // Start a transaction
    await complaintsPool.query('BEGIN');
    
    // Update the complaint status
    const result = await complaintsPool.query(
      'UPDATE complaints SET status = $1, admin_notes = $2, updated_at = NOW() WHERE id = $3 RETURNING *',
      [status, admin_notes, id]
    );
    
    if (result.rows.length === 0) {
      await complaintsPool.query('ROLLBACK');
      return res.status(404).json({ success: false, message: 'Complaint not found.' });
    }
    
    // Add entry to status history table
    try {
      // Check if the table exists, if not create it
      await complaintsPool.query(`
        CREATE TABLE IF NOT EXISTS complaint_status_history (
          id SERIAL PRIMARY KEY,
          complaint_id INTEGER REFERENCES complaints(id) ON DELETE CASCADE,
          status VARCHAR(50) NOT NULL,
          admin_notes TEXT,
          created_at TIMESTAMP DEFAULT NOW(),
          timestamp TIMESTAMP DEFAULT NOW()
        )
      `);
      
      // Check if timestamp column exists, if not add it
      try {
        await complaintsPool.query(`
          DO $$
          BEGIN
            IF NOT EXISTS (SELECT 1 FROM information_schema.columns 
                          WHERE table_name='complaint_status_history' AND column_name='timestamp') THEN
              ALTER TABLE complaint_status_history ADD COLUMN timestamp TIMESTAMP DEFAULT NOW();
              -- Update existing records to have timestamp match created_at
              UPDATE complaint_status_history SET timestamp = created_at WHERE timestamp IS NULL;
            END IF;
          END
          $$;
        `);
      } catch (alterErr) {
        console.error('Error checking/adding timestamp column:', alterErr);
        // Continue even if column check/add fails
      }
      
      // Insert the status update into history
      await complaintsPool.query(
        'INSERT INTO complaint_status_history (complaint_id, status, admin_notes, timestamp) VALUES ($1, $2, $3, NOW())',
        [id, status, admin_notes]
      );
    } catch (historyErr) {
      logger.error(`Error updating status history: ${historyErr.message}`);
      // Continue even if history update fails
    }
    
    // Commit the transaction
    await complaintsPool.query('COMMIT');
    
    res.json({ 
      success: true, 
      message: 'Complaint status updated successfully!',
      complaint: result.rows[0]
    });
  } catch (err) {
    await complaintsPool.query('ROLLBACK');
    logger.error(`Error updating complaint status: ${err.message}`);
    res.status(500).json({ success: false, message: 'Failed to update complaint status.' });
  }
});

// Get user by student_id endpoint
app.get('/api/user', async (req, res) => {
  const { student_id } = req.query;
  
  try {
    const result = await mainPool.query('SELECT * FROM users WHERE student_id = $1', [student_id]);
    
    if (result.rows.length === 0) {
      return res.status(404).json({ success: false, message: 'User not found.' });
    }
    
    res.json({ 
      success: true, 
      user: result.rows[0]
    });
  } catch (err) {
    logger.error(`Error fetching user: ${err.message}`);
    res.status(500).json({ success: false, message: 'Failed to fetch user.' });
  }
});

// Get all users endpoint (admin only)
app.get('/api/users', async (req, res) => {
  try {
    const result = await mainPool.query('SELECT id, name, student_id, role, created_at FROM users ORDER BY id');
    
    res.json({ 
      success: true, 
      users: result.rows
    });
  } catch (err) {
    logger.error('Error fetching users:', err);
    res.status(500).json({ success: false, message: 'Failed to fetch users.' });
  }
});

// Update user endpoint (admin only)
app.put('/api/users/:id', async (req, res) => {
  try {
    const { id } = req.params;
    const { name, student_id, role, password } = req.body;
    
    // Validate id parameter
    if (!id || isNaN(parseInt(id))) {
      return res.status(400).json({ success: false, message: 'Invalid user ID' });
    }

    // Start building the query
    let query = 'UPDATE users SET ';
    const queryParams = [];
    const updates = [];
    let paramCounter = 1;

    // Add fields to update
    if (name) {
      updates.push(`name = $${paramCounter}`);
      queryParams.push(name);
      paramCounter++;
    }

    if (student_id) {
      updates.push(`student_id = $${paramCounter}`);
      queryParams.push(student_id);
      paramCounter++;
    }

    if (role) {
      updates.push(`role = $${paramCounter}`);
      queryParams.push(role);
      paramCounter++;
    }

    // Handle password update if provided
    if (password) {
      const hashedPassword = await bcrypt.hash(password, 10);
      updates.push(`password = $${paramCounter}`);
      queryParams.push(hashedPassword);
      paramCounter++;
    }

    // If no fields to update
    if (updates.length === 0) {
      return res.status(400).json({ success: false, message: 'No fields to update' });
    }

    // Complete the query
    query += updates.join(', ');
    query += ` WHERE id = $${paramCounter} RETURNING id, name, student_id, role`;
    queryParams.push(id);

    // Execute the query
    const result = await mainPool.query(query, queryParams);

    if (result.rows.length === 0) {
      return res.status(404).json({ success: false, message: 'User not found' });
    }

    res.json({ success: true, user: result.rows[0] });
  } catch (error) {
    console.error('Error updating user:', error);
    res.status(500).json({ success: false, message: 'Server error' });
  }
});

// Reset user password endpoint (admin only)
app.put('/api/users/:id/reset-password', async (req, res) => {
  try {
    const { id } = req.params;
    const { password } = req.body;

    if (!password) {
      return res.status(400).json({ success: false, message: 'Password is required' });
    }

    // Hash the new password
    const hashedPassword = await bcrypt.hash(password, 10);

    // Update the password
    const result = await mainPool.query(
      'UPDATE users SET password = $1 WHERE id = $2 RETURNING id, name',
      [hashedPassword, id]
    );

    if (result.rows.length === 0) {
      return res.status(404).json({ success: false, message: 'User not found' });
    }

    res.json({ success: true, message: 'Password reset successfully' });
  } catch (error) {
    console.error('Error resetting password:', error);
    res.status(500).json({ success: false, message: 'Server error' });
  }
});

// Delete user endpoint (admin only)
app.delete('/api/users/:id', async (req, res) => {
  try {
    const { id } = req.params;
    
    // Validate id parameter
    if (!id || isNaN(parseInt(id))) {
      return res.status(400).json({ success: false, message: 'Invalid user ID' });
    }
    
    // Check if user has complaints and delete them first
    await complaintsPool.query('DELETE FROM complaints WHERE user_id = $1', [id]);

    // Delete the user
    const result = await mainPool.query('DELETE FROM users WHERE id = $1 RETURNING id, name', [id]);

    if (result.rows.length === 0) {
      return res.status(404).json({ success: false, message: 'User not found' });
    }

    res.json({ success: true, message: 'User deleted successfully' });
  } catch (error) {
    console.error('Error deleting user:', error);
    res.status(500).json({ success: false, message: 'Server error' });
  }
});

// Get complaint statistics endpoint
app.get('/api/stats', async (req, res) => {
  try {
    const totalComplaints = await complaintsPool.query('SELECT COUNT(*) FROM complaints');
    const totalUsers = await mainPool.query('SELECT COUNT(*) FROM users WHERE role != \'admin\'');
    const pendingComplaints = await complaintsPool.query('SELECT COUNT(*) FROM complaints WHERE status = \'pending\'');
    const resolvedComplaints = await complaintsPool.query('SELECT COUNT(*) FROM complaints WHERE status = \'resolved\'');
    
    res.json({
      success: true,
      stats: {
        totalComplaints: parseInt(totalComplaints.rows[0].count),
        totalUsers: parseInt(totalUsers.rows[0].count),
        pendingComplaints: parseInt(pendingComplaints.rows[0].count),
        resolvedComplaints: parseInt(resolvedComplaints.rows[0].count)
      }
    });
  } catch (err) {
    console.error('Error fetching statistics:', err);
    res.status(500).json({ success: false, message: 'Failed to fetch statistics.' });
  }
});

// Delete complaint endpoint (admin only)
app.delete('/api/complaints/:id', async (req, res) => {
  const { id } = req.params;
  
  // Validate parameters
  if (!id || isNaN(parseInt(id))) {
    return res.status(400).json({ success: false, message: 'Invalid complaint ID' });
  }
  
  try {
    // Start a transaction
    await complaintsPool.query('BEGIN');
    
    // Delete status history first (if exists)
    try {
      await complaintsPool.query('DELETE FROM complaint_status_history WHERE complaint_id = $1', [id]);
    } catch (historyErr) {
      console.log('No status history to delete or table does not exist');
      // Continue even if this fails
    }
    
    // Delete the complaint
    const result = await complaintsPool.query('DELETE FROM complaints WHERE id = $1 RETURNING id', [id]);
    
    if (result.rows.length === 0) {
      await complaintsPool.query('ROLLBACK');
      return res.status(404).json({ success: false, message: 'Complaint not found' });
    }
    
    // Commit the transaction
    await complaintsPool.query('COMMIT');
    
    res.json({ success: true, message: 'Complaint deleted successfully' });
  } catch (err) {
    await complaintsPool.query('ROLLBACK');
    console.error('Error deleting complaint:', err);
    res.status(500).json({ success: false, message: 'Failed to delete complaint' });
  }
});

// Track complaint endpoint
app.get('/api/complaints/:id/track', async (req, res) => {
  const { id } = req.params;
  const { student_id } = req.query;
  
  // Validate parameters
  if (!id || isNaN(parseInt(id))) {
    return res.status(400).json({ success: false, message: 'Invalid complaint ID' });
  }
  
  if (!student_id) {
    return res.status(400).json({ success: false, message: 'Student ID is required' });
  }
  
  try {
    // Get the user ID from student_id
    const userResult = await mainPool.query('SELECT id FROM users WHERE student_id = $1', [student_id]);
    
    if (userResult.rows.length === 0) {
      return res.status(404).json({ success: false, message: 'User not found' });
    }
    
    const userId = userResult.rows[0].id;
    
    // Get the complaint from complaints database
    const complaintResult = await complaintsPool.query(
      'SELECT * FROM complaints WHERE id = $1',
      [id]
    );
    
    if (complaintResult.rows.length === 0) {
      return res.status(404).json({ success: false, message: 'Complaint not found' });
    }
    
    const complaint = complaintResult.rows[0];
    
    // Check if the user is authorized to view this complaint
    if (complaint.user_id.toString() !== userId.toString()) {
      return res.status(403).json({ success: false, message: 'You are not authorized to view this complaint' });
    }
    
    // Get status history if available
    let statusHistory = [];
    try {
      const historyResult = await complaintsPool.query(
        'SELECT status, admin_notes as note, created_at as timestamp FROM complaint_status_history WHERE complaint_id = $1 ORDER BY created_at ASC',
        [id]
      );
      statusHistory = historyResult.rows;
      console.log(`Retrieved ${statusHistory.length} status history entries for complaint ${id}:`, statusHistory);
      
      // If no status history is found, create an initial entry
      if (statusHistory.length === 0) {
        console.log('No status history found, creating initial entry');
        await complaintsPool.query(
          'INSERT INTO complaint_status_history (complaint_id, status) VALUES ($1, $2) RETURNING id',
          [id, complaint.status]
        );
        
        // Fetch the newly created history
        const newHistoryResult = await complaintsPool.query(
          'SELECT status, admin_notes as note, created_at as timestamp FROM complaint_status_history WHERE complaint_id = $1 ORDER BY created_at ASC',
          [id]
        );
        statusHistory = newHistoryResult.rows;
      }
    } catch (historyErr) {
      console.error('Status history not available:', historyErr);
    }
    
    // Add status history to the complaint object
    complaint.status_history = statusHistory;
    
    // Add attachment URL if attachment path exists
    if (complaint.attachment_path) {
      const serverUrl = req.protocol + '://' + req.get('host');
      complaint.attachment_url = serverUrl + (complaint.attachment_path.startsWith('/') ? '' : '/') + complaint.attachment_path;
    }
    
    res.json({ success: true, complaint });
  } catch (err) {
    console.error('Error tracking complaint:', err);
    res.status(500).json({ success: false, message: 'Failed to track complaint' });
  }
});

// Delete complaint endpoint (admin only)
app.delete('/api/complaints/:id', async (req, res) => {
  try {
    const { id } = req.params;
    
    // Validate id parameter
    if (!id || isNaN(parseInt(id))) {
      return res.status(400).json({ success: false, message: 'Invalid complaint ID' });
    }
    
    // Start a transaction
    await complaintsPool.query('BEGIN');
    
    try {
      // Delete status history first (if exists)
      await complaintsPool.query('DELETE FROM complaint_status_history WHERE complaint_id = $1', [id]);
      
      // Delete the complaint
      const result = await complaintsPool.query('DELETE FROM complaints WHERE id = $1 RETURNING id', [id]);
      
      if (result.rows.length === 0) {
        await complaintsPool.query('ROLLBACK');
        return res.status(404).json({ success: false, message: 'Complaint not found' });
      }
      
      await complaintsPool.query('COMMIT');
      res.json({ success: true, message: 'Complaint deleted successfully' });
    } catch (err) {
      await complaintsPool.query('ROLLBACK');
      throw err;
    }
  } catch (error) {
    console.error('Error deleting complaint:', error);
    res.status(500).json({ success: false, message: 'Server error' });
  }
});

// Update complaint admin notes endpoint
app.put('/api/complaints/:id/notes', async (req, res) => {
  try {
    const { id } = req.params;
    const { admin_notes } = req.body;
    
    // Validate id parameter
    if (!id || isNaN(parseInt(id))) {
      return res.status(400).json({ success: false, message: 'Invalid complaint ID' });
    }
    
    // Update the complaint admin notes
    const result = await complaintsPool.query(
      'UPDATE complaints SET admin_notes = $1, updated_at = NOW() WHERE id = $2 RETURNING id',
      [admin_notes, id]
    );
    
    if (result.rows.length === 0) {
      return res.status(404).json({ success: false, message: 'Complaint not found' });
    }
    
    res.json({ success: true, message: 'Admin notes updated successfully' });
  } catch (error) {
    console.error('Error updating admin notes:', error);
    res.status(500).json({ success: false, message: 'Server error' });
  }
});


// Error handling middleware
app.use((err, req, res, next) => {
  console.error('Unhandled error:', err.stack);
  res.status(500).json({ success: false, message: 'Internal server error' });
});

// Handle 404 errors
app.use((req, res) => {
  res.status(404).json({ success: false, message: 'Not found' });
});

// Initialize the complaints database tables
async function initializeComplaintsTables() {
  if (!complaintsPool) {
    logger.error('Cannot initialize tables: complaintsPool is not initialized');
    return false;
  }
  
  try {
    // Create users table if it doesn't exist
    await complaintsPool.query(`
      CREATE TABLE IF NOT EXISTS users (
        id SERIAL PRIMARY KEY,
        name VARCHAR(100) NOT NULL,
        student_id VARCHAR(8) UNIQUE NOT NULL,
        password VARCHAR(255) NOT NULL,
        email VARCHAR(100),
        role VARCHAR(20) DEFAULT 'student',
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
      )
    `);
    logger.info('Users table initialized');
    
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
    
    // Check if attachment_path column exists, add it if it doesn't
    try {
      // Check if column exists
      const columnCheck = await complaintsPool.query(`
        SELECT column_name 
        FROM information_schema.columns 
        WHERE table_name='complaints' AND column_name='attachment_path'
      `);
      
      // If column doesn't exist, add it
      if (columnCheck.rows.length === 0) {
        console.log('Adding attachment_path column to complaints table');
        await complaintsPool.query(`
          ALTER TABLE complaints 
          ADD COLUMN attachment_path TEXT
        `);
      }
    } catch (columnErr) {
      console.error('Error checking/adding attachment_path column:', columnErr);
    }
    
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
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
      )
    `);
    
    // Fix the sequence for the complaint_status_history table
    await complaintsPool.query(`
      SELECT setval('complaint_status_history_id_seq', COALESCE((SELECT MAX(id) FROM complaint_status_history), 0) + 1, false);
    `);
    
    logger.info('Complaints database tables initialized successfully');
    return true;
  } catch (err) {
    console.error('Error initializing complaints database tables:', err);
    return false;
  }
}

// Initialize the database and start the server
async function startServer() {
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
    
    // Start the server
    app.listen(3000, () => {
      logger.info('Server running on port 3000');
    });
  } catch (err) {
    logger.error('Server initialization error:', err);
    process.exit(1);
  }
}

// WebSocket server for real-time updates
const WebSocket = require('ws');
let wss;

// Function to initialize WebSocket server
function initializeWebSocketServer() {
  try {
    wss = new WebSocket.Server({ port: 3001 });
    logger.info('WebSocket server started on port 3001');
  } catch (error) {
    if (error.code === 'EADDRINUSE') {
      logger.warn('Port 3001 is already in use. WebSocket server not started.');
      logger.info('Application will continue without WebSocket functionality.');
      return; // Exit the function without initializing WebSocket
    } else {
      logger.error('Failed to start WebSocket server:', error);
      return; // Exit the function without initializing WebSocket
    }
  }
  
  wss.on('connection', (ws) => {
    logger.info('New WebSocket connection established');
    ws.isAlive = true;
    
    // Handle authentication and message routing
    ws.on('message', (message) => {
      try {
        logger.debug('WebSocket message received:', message.toString());
        const data = JSON.parse(message);
        
        // Handle authentication
        if (data.type === 'auth') {
          ws.userId = data.userId;
          ws.role = data.role;
          logger.debug(`WebSocket authenticated: ${ws.role} ${ws.userId}`);
        }
        
        // Handle subscription to specific complaint updates
        if (data.type === 'subscribe' && data.complaintId) {
          ws.complaintId = data.complaintId;
          logger.debug(`Client subscribed to updates for complaint: ${data.complaintId}`);
        }
        
        // Handle ping messages to keep connection alive
        if (data.type === 'ping') {
          ws.isAlive = true;
          ws.send(JSON.stringify({ type: 'pong' }));
        }
      } catch (err) {
        logger.error('Error processing WebSocket message:', err);
      }
    });
    
    ws.on('close', () => {
      logger.debug('WebSocket connection closed');
      ws.isAlive = false;
    });
    
    ws.on('error', (error) => {
      logger.error('WebSocket client error:', error);
    });
    
    // Send initial connection confirmation
    ws.send(JSON.stringify({ 
      type: 'connection_established',
      timestamp: new Date().toISOString()
    }));
  });
  
  // Handle WebSocket server errors
  wss.on('error', (error) => {
    logger.error('WebSocket server error:', error);
  });
  
  // Set up a heartbeat interval to detect dead connections
  const interval = setInterval(() => {
    wss.clients.forEach((ws) => {
      if (ws.isAlive === false) {
        logger.debug('Terminating inactive WebSocket connection');
        return ws.terminate();
      }
      
      ws.isAlive = false;
      ws.send(JSON.stringify({ type: 'heartbeat' }));
    });
  }, 30000); // Check every 30 seconds
  
  wss.on('close', () => {
    clearInterval(interval);
  });
}

// Function to broadcast status updates to connected clients
function broadcastStatusUpdate(complaintId, status, userId) {
  if (!wss) return;
  
  logger.info(`Broadcasting status update for complaint ${complaintId} to user ${userId} and admins. Status: ${status}`);
  
  wss.clients.forEach((client) => {
    // Send to the complaint owner or admins
    if (client.readyState === WebSocket.OPEN) {
      logger.debug(`Checking client: userId=${client.userId}, role=${client.role}`);
      
      if (client.userId === userId.toString() || client.role === 'admin') {
        logger.debug(`Sending status update to client: ${client.userId || 'unknown'} (${client.role || 'unknown role'})`);
        client.send(JSON.stringify({
          type: 'status_update',
          complaint: {
            id: complaintId,
            status: status,
            updated_at: new Date().toISOString()
          }
        }));
      }
    }
  });
}

// Update the status update endpoint to broadcast changes
app.put('/api/complaints/:id/status', async (req, res) => {
  const { id } = req.params;
  const { status, admin_notes } = req.body;
  
  console.log(`Updating complaint ${id} status to ${status} with notes: ${admin_notes}`);
  
  try {
    // Start a transaction
    await complaintsPool.query('BEGIN');
    
    // Get the user_id for the complaint first
    const userIdResult = await complaintsPool.query(
      'SELECT user_id FROM complaints WHERE id = $1',
      [id]
    );
    
    if (userIdResult.rows.length === 0) {
      await complaintsPool.query('ROLLBACK');
      return res.status(404).json({ success: false, message: 'Complaint not found.' });
    }
    
    const userId = userIdResult.rows[0].user_id;
    console.log(`Found complaint ${id} belonging to user ${userId}`);
    
    // Update the complaint status
    const result = await complaintsPool.query(
      'UPDATE complaints SET status = $1, admin_notes = $2, updated_at = NOW() WHERE id = $3 RETURNING *',
      [status, admin_notes, id]
    );
    
    // Add entry to status history table
    try {
      // Check if the table exists, if not create it
      await complaintsPool.query(`
        CREATE TABLE IF NOT EXISTS complaint_status_history (
          id SERIAL PRIMARY KEY,
          complaint_id INTEGER REFERENCES complaints(id) ON DELETE CASCADE,
          status VARCHAR(50) NOT NULL,
          admin_notes TEXT,
          updated_at TIMESTAMP DEFAULT NOW()
        )
      `);
      
      // Insert the status update into history
      const historyResult = await complaintsPool.query(
        'INSERT INTO complaint_status_history (complaint_id, status, admin_notes) VALUES ($1, $2, $3) RETURNING id, created_at',
        [id, status, admin_notes]
      );
      
      console.log(`Added status history entry ${historyResult.rows[0].id} for complaint ${id} at ${historyResult.rows[0].created_at}`);
    } catch (historyErr) {
      console.error('Error updating status history:', historyErr);
      // Continue even if history update fails
    }
    
    // Commit the transaction
    await complaintsPool.query('COMMIT');
    
    // Broadcast the status update via WebSocket
    console.log(`Broadcasting status update for complaint ${id}`);
    broadcastStatusUpdate(id, status, userId);
    
    res.json({ 
      success: true, 
      message: 'Complaint status updated successfully!',
      complaint: result.rows[0]
    });
  } catch (err) {
    await complaintsPool.query('ROLLBACK');
    console.error('Error updating complaint status:', err);
    res.status(500).json({ success: false, message: 'Failed to update complaint status.' });
  }
});

// Start the server
startServer();

// Initialize WebSocket server after HTTP server starts
initializeWebSocketServer();