# Project Setup

This guide will help you set up the project and get it running on your local machine.

## Backend Setup

1. Install the required Python packages:
   - Navigate to the project directory where the `requirements.txt` file is located.
   - Run the following command to install the necessary Python dependencies:

   ```bash
   pip install -r requirements.txt
   ```

2. Start the backend server:
   - After installing the requirements, run the `server.py` file to start the backend server:

   ```bash
   python server.py
   ```

## Frontend Setup

1. Navigate to the `project` folder where the frontend code is located:

   ```bash
   cd project
   ```

2. Install the required Node.js dependencies:
   - Run the following command to install the necessary packages for the frontend:

   ```bash
   npm install
   ```

3. Start the frontend server:
   - Once the dependencies are installed, run the following command to start the frontend:

   ```bash
   npm run start
   ```

Your backend and frontend servers should now be running. Open your browser and navigate to `http://localhost:5173` (or another port if configured) to access the application.

## Notes

- Ensure you have Python 3.x, ffmpeg and Node.js installed on your machine.
- If you encounter any issues, check the dependencies in `requirements.txt` and `package.json` for any additional configuration.
