#!/usr/bin/env python3
import sys
import os
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from app.api.auth import generate_api_key, get_hashed_api_key

def main():
    # Generate new API key and hash
    api_key = generate_api_key()
    hashed_key = get_hashed_api_key(api_key)

    print("\n=== Generated API Key Pair ===")
    print(f"\nAPI Key (for clients):\n{api_key}")
    print(f"\nHashed Key (for API server):\n{hashed_key}")

    # Ask if user wants to save to .env files
    save = input("\nWould you like to save these to .env files? (y/n): ").lower().strip()

    if save == 'y':
        # Save to API .env
        env_path = Path(__file__).parent.parent / '.env'
        try:
            # Read existing .env content
            if env_path.exists():
                with open(env_path, 'r') as f:
                    lines = f.readlines()
                # Remove existing HASHED_API_KEY if present
                lines = [l for l in lines if not l.startswith('HASHED_API_KEY=')]
            else:
                lines = []

            # Add new hashed key
            lines.append(f'\nHASHED_API_KEY={hashed_key}\n')

            # Write back to file
            with open(env_path, 'w') as f:
                f.writelines(lines)
            print(f"\n✅ Saved hashed key to {env_path}")
        except Exception as e:
            print(f"\n❌ Error saving to API .env: {e}")

        # Save to site .env
        site_env_path = Path(__file__).parent.parent / 'app' / 'site' / '.env'
        try:
            site_env_path.parent.mkdir(exist_ok=True)

            # Read existing .env content
            if site_env_path.exists():
                with open(site_env_path, 'r') as f:
                    lines = f.readlines()
                # Remove existing API_KEY if present
                lines = [l for l in lines if not l.startswith('API_KEY=')]
            else:
                lines = []

            # Add new API key
            lines.append(f'\nAPI_KEY={api_key}\n')

            # Write back to file
            with open(site_env_path, 'w') as f:
                f.writelines(lines)
            print(f"✅ Saved API key to {site_env_path}")
        except Exception as e:
            print(f"❌ Error saving to site .env: {e}")

        print("\n⚠️  Important: Keep your API key secure and never commit it to version control!")

    print("\nDone! 🎉")

if __name__ == "__main__":
    main()
