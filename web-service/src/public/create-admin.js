const { Pool } = require('pg');
const bcrypt = require('bcrypt');

const pool = new Pool({
  user: 'postgres',
  host: 'localhost',
  database: 'postgres',
  password: 'anjali@12',
  port: 5432,
});

async function createAdminUser() {
  try {
    // Admin user details
    const adminUser = {
      name: 'Admin User',
      student_id: '20230000', // Special ID for admin
      password: 'admin123',   // Default password
      role: 'admin'
    };

    // Check if admin already exists
    const existingAdmin = await pool.query(
      'SELECT * FROM users WHERE student_id = $1',
      [adminUser.student_id]
    );

    if (existingAdmin.rows.length > 0) {
      console.log('Admin user already exists!');
      return;
    }

    // Hash the password
    const saltRounds = 10;
    const hash = await bcrypt.hash(adminUser.password, saltRounds);

    // Insert admin user
    await pool.query(
      'INSERT INTO users (name, student_id, password, role) VALUES ($1, $2, $3, $4)',
      [adminUser.name, adminUser.student_id, hash, adminUser.role]
    );

    console.log('Admin user created successfully!');
    console.log('Login credentials:');
    console.log('Student ID:', adminUser.student_id);
    console.log('Password:', adminUser.password);
  } catch (err) {
    console.error('Error creating admin user:', err);
  } finally {
    pool.end();
  }
}

createAdminUser();