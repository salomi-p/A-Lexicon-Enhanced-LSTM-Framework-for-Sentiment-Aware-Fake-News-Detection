-- Create Database
CREATE DATABASE IF NOT EXISTS fake_news_classify;
USE fake_news_classify;

-- Admin Table
CREATE TABLE IF NOT EXISTS admin (
    id INT AUTO_INCREMENT PRIMARY KEY,
    username VARCHAR(50) NOT NULL UNIQUE,
    password VARCHAR(255) NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- Insert default admin (password: admin123)
INSERT INTO admin (username, password) VALUES ('admin', 'admin123');

-- User Registration Table
CREATE TABLE IF NOT EXISTS register (
    id INT AUTO_INCREMENT PRIMARY KEY,
    name VARCHAR(100) NOT NULL,
    mobile VARCHAR(20),
    email VARCHAR(100) NOT NULL UNIQUE,
    uname VARCHAR(50) NOT NULL UNIQUE,
    pass VARCHAR(255) NOT NULL,
    dob DATE,
    location VARCHAR(100),
    profession VARCHAR(100),
    aadhar VARCHAR(20),
    photo INT DEFAULT 0,
    status INT DEFAULT 0,
    dstatus INT DEFAULT 0 COMMENT '0=active, 1=blocked',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- User Posts Table
CREATE TABLE IF NOT EXISTS user_post (
    id INT AUTO_INCREMENT PRIMARY KEY,
    uname VARCHAR(50) NOT NULL,
    title VARCHAR(255),
    text_post TEXT NOT NULL,
    photo VARCHAR(255),
    rdate VARCHAR(20),
    status INT DEFAULT 0 COMMENT '0=not classified, 1=fake, 2=normal, 3=true',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (uname) REFERENCES register(uname) ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- Create indexes for better performance
CREATE INDEX idx_uname ON register(uname);
CREATE INDEX idx_email ON register(email);
CREATE INDEX idx_post_uname ON user_post(uname);
CREATE INDEX idx_post_status ON user_post(status);
CREATE INDEX idx_dstatus ON register(dstatus);

-- Display table structures
DESCRIBE admin;
DESCRIBE register;
DESCRIBE user_post;

-- Show tables
SHOW TABLES;

-- Success message
SELECT 'Database schema created successfully!' AS Message;
