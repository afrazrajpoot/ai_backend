-- CreateEnum
CREATE TYPE "CompanyApprovalStatus" AS ENUM ('PENDING', 'APPROVED', 'REJECTED');

-- CreateEnum
CREATE TYPE "SubscriptionTier" AS ENUM ('FREE', 'PAID');

-- CreateEnum
CREATE TYPE "PaymentStatus" AS ENUM ('PENDING', 'PAID', 'OVERDUE');

-- AlterTable
ALTER TABLE "Company" ADD COLUMN     "approvalStatus" "CompanyApprovalStatus" NOT NULL DEFAULT 'PENDING',
ADD COLUMN     "currentSeats" INTEGER NOT NULL DEFAULT 0,
ADD COLUMN     "maxSeats" INTEGER NOT NULL DEFAULT 5,
ADD COLUMN     "notes" TEXT,
ADD COLUMN     "paymentStatus" "PaymentStatus" NOT NULL DEFAULT 'PENDING',
ADD COLUMN     "subscriptionEndDate" TIMESTAMP(3),
ADD COLUMN     "subscriptionStartDate" TIMESTAMP(3),
ADD COLUMN     "subscriptionTier" "SubscriptionTier" NOT NULL DEFAULT 'FREE';
