/*
  Warnings:

  - You are about to drop the column `position` on the `Department` table. All the data in the column will be lost.
  - You are about to drop the column `userId` on the `Department` table. All the data in the column will be lost.
  - The `position` column on the `User` table would be dropped and recreated. This will lead to data loss if there is data in the column.
  - Added the required column `updatedAt` to the `Department` table without a default value. This is not possible if the table is not empty.

*/
-- AlterTable
ALTER TABLE "Department" DROP COLUMN "position",
DROP COLUMN "userId",
ADD COLUMN     "createdAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
ADD COLUMN     "updatedAt" TIMESTAMP(3) NOT NULL,
ALTER COLUMN "name" SET DATA TYPE TEXT;

-- AlterTable
ALTER TABLE "User" DROP COLUMN "position",
ADD COLUMN     "position" TEXT[];
