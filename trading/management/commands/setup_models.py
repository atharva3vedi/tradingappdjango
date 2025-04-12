from django.core.management.base import BaseCommand
from trading.model_manager import model_manager
import logging

logger = logging.getLogger(__name__)

class Command(BaseCommand):
    help = 'Setup and update trading models'

    def add_arguments(self, parser):
        parser.add_argument(
            '--force',
            action='store_true',
            help='Force update all models even if they exist',
        )
        parser.add_argument(
            '--category',
            type=str,
            help='Update only specific category (tech, finance, healthcare, energy)',
        )

    def handle(self, *args, **options):
        force = options['force']
        category = options['category']

        if category and category not in model_manager.categories:
            self.stderr.write(f"Invalid category: {category}")
            self.stderr.write(f"Valid categories are: {', '.join(model_manager.categories.keys())}")
            return

        try:
            if force:
                self.stdout.write("Forcing update of all models...")
                model_manager.update_models()
            elif category:
                self.stdout.write(f"Updating {category} model...")
                model_manager._train_new_model(category)
            else:
                self.stdout.write("Checking and updating models as needed...")
                model_manager.update_models()

            self.stdout.write(self.style.SUCCESS("Model setup/update completed successfully"))
        except Exception as e:
            self.stderr.write(self.style.ERROR(f"Error during model setup/update: {str(e)}"))
            logger.error(f"Error during model setup/update: {str(e)}", exc_info=True) 