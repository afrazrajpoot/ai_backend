/*
  Warnings:

  - Added the required column `position` to the `Department` table without a default value. This is not possible if the table is not empty.

*/
-- AlterTable
ALTER TABLE "Department" ADD COLUMN     "position" JSONB NOT NULL;
